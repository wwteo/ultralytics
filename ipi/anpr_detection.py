import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import os
import csv
import logging
from queue import Queue
import threading
from collections import defaultdict
import torch
import paddle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify PaddlePaddle installation
try:
    paddle.utils.run_check()
    if not paddle.device.is_compiled_with_cuda():
        logger.warning("PaddlePaddle not compiled with CUDA. Check installation and CUDA/cuDNN setup.")
except Exception as e:
    logger.error("PaddlePaddle installation check failed: %s", e)
    exit(1)

# Verify GPU availability
logger.info("PyTorch GPU Available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("PyTorch Device: %s", torch.cuda.get_device_name(0))
logger.info("PaddlePaddle GPU Available: %s", paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    logger.info("Paddle Device: %s", paddle.device.get_device())
else:
    logger.warning("PaddlePaddle GPU unavailable, using CPU. Check CUDA/cuDNN installation.")

# Initialize YOLO and PaddleOCR with documented settings
try:
    model = YOLO(r".\best.pt")
    ocr = PaddleOCR(lang='en',
                    use_doc_orientation_classify=True,  # Enables document orientation classification
                    use_doc_unwarping=True,            # Enables text image rectification
                    use_textline_orientation=True)     # Enables text line orientation classification
    logger.info("YOLO and PaddleOCR initialized successfully")
except Exception as e:
    logger.error("Error initializing models: %s", e)
    exit(1)

# Preprocess plate image with simplified technique
def preprocess_plate(plate_crop, frame_count):
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=5)  # Basic contrast adjustment
        cv2.imwrite(f"preprocessed_plate_{frame_count}.jpg", gray)  # Optional debug save
        return gray
    except Exception as e:
        logger.error("Error in preprocessing: %s", e)
        return None

# Resize small plates
def resize_plate(plate_crop):
    try:
        min_height, min_width = 100, 300
        if plate_crop.shape[0] < min_height or plate_crop.shape[1] < min_width:
            scale = max(min_height / plate_crop.shape[0], min_width / plate_crop.shape[1])
            new_size = (int(plate_crop.shape[1] * scale), int(plate_crop.shape[0] * scale))
            resized = cv2.resize(plate_crop, new_size, interpolation=cv2.INTER_CUBIC)
            return resized
        return plate_crop
    except Exception as e:
        logger.error("Error in resizing: %s", e)
        return plate_crop

# UK plate validation with post-processing
def validate_uk_plate(text):
    text = text.upper().replace('O', '0').replace('I', '1').replace('S', '5')
    filtered_text = ''.join(c for c in text if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    pattern = r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$'
    return re.match(pattern, filtered_text), filtered_text

# Aggregate OCR results with fallback
def aggregate_plate_texts(plate_texts, min_confidence=0.7, min_occurrences=5):
    if not plate_texts:
        return ""
    text_counts = defaultdict(int)
    for text, conf in plate_texts:
        is_valid, filtered_text = validate_uk_plate(text)
        if is_valid and conf >= min_confidence:
            text_counts[filtered_text] += 1
    if not text_counts:
        if plate_texts:
            text, conf = max(plate_texts, key=lambda x: x[1])
            is_valid, filtered_text = validate_uk_plate(text)
            return filtered_text if is_valid and conf >= 0.3 else ""
        return ""
    most_common = max(text_counts.items(), key=lambda x: x[1], default=("", 0))
    if most_common[1] >= min_occurrences:
        return most_common[0]
    return ""

# Process a single frame with confidence filtering
def process_frame(frame_queue, results_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        try:
            results = model.track(frame, persist=True, classes=[0], conf=0.75)
            results_queue.put(results)
        except Exception as e:
            logger.error("Error in frame processing: %s", e)
            results_queue.put(None)
        frame_queue.task_done()

# Process a single video
def process_video(video_path, output_dir, csv_file_path, ocr_interval=45):
    logger.info("Processing video: %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video at %s", video_path)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_dir, os.path.basename(video_path).replace(".mp4", "_processed.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["frame_number", "license_text", "confidence"])
        writer.writeheader()

        frame_queue = Queue(maxsize=5)
        results_queue = Queue()
        threads = [threading.Thread(target=process_frame, args=(frame_queue, results_queue)) for _ in range(2)]
        for t in threads:
            t.start()

        plate_tracker = defaultdict(list)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.resize(frame, (1920, 1080))

            if frame_count % 2 == 0:
                frame_queue.put(frame)
                results = results_queue.get()
                if results is None or not results[0].boxes:
                    out.write(frame)
                    continue

                plates = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confidences):
                        if conf < 0.6:
                            continue
                        x_min, y_min, x_max, y_max = map(int, box)
                        padding = 10
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(frame.shape[1], x_max + padding)
                        y_max = min(frame.shape[0], y_max + padding)
                        plate_crop = frame[y_min:y_max, x_min:x_max]
                        plates.append((box, plate_crop))

                plate_crops = []
                plate_keys = []
                for i, (box, plate_crop) in enumerate(plates):
                    x_min, y_min, x_max, y_max = map(int, box)
                    box_key = (x_min, y_min, x_max, y_max)
                    plate_crop = resize_plate(plate_crop)
                    preprocessed_plate = preprocess_plate(plate_crop, frame_count)
                    if preprocessed_plate is None:
                        continue
                    plate_crops.append(preprocessed_plate)
                    plate_keys.append((i, box_key, plate_crop))

                if plate_crops and (frame_count % ocr_interval == 0 or any(not plate_tracker[key[1]] for key in plate_keys)):
                    images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in plate_crops]  # Convert to RGB for PaddleOCR
                    try:
                        result = ocr.predict(images)
                        for i, (idx, box_key, orig_crop) in enumerate(plate_keys):
                            text = ""
                            conf = 0.0
                            if result and len(result) > i and result[i] and result[i][0] and result[i][0][1]:
                                text = result[i][0][1][0].strip().upper()
                                conf = result[i][0][1][1]
                                logger.info("OCR Result - Frame %d, Plate %d: Text: %s, Confidence: %.2f", frame_count, idx+1, text, conf)
                            else:
                                # Fallback with original crop
                                fallback_result = ocr.predict([cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB)])
                                if fallback_result and len(fallback_result) > 0 and fallback_result[0] and fallback_result[0][0] and fallback_result[0][0][1]:
                                    text = fallback_result[0][0][1][0].strip().upper()
                                    conf = fallback_result[0][0][1][1]
                                    logger.info("Fallback OCR Result - Frame %d, Plate %d: Text: %s, Confidence: %.2f", frame_count, idx+1, text, conf)
                            plate_tracker[box_key].append((text, conf))
                    except Exception as e:
                        logger.error("Video %s, Frame %d: OCR failed - %s", os.path.basename(video_path), frame_count, str(e))
                        for i, _, orig_crop in plate_keys:
                            cv2.imwrite(f"failed_plate_{frame_count}_{i}.jpg", orig_crop)

                for i, (box, plate_crop) in enumerate(plates):
                    x_min, y_min, x_max, y_max = map(int, box)
                    box_key = (x_min, y_min, x_max, y_max)
                    aggregated_text = aggregate_plate_texts(plate_tracker[box_key])
                    if aggregated_text:
                        is_valid, plate_text = validate_uk_plate(aggregated_text)
                        display_text = f"{plate_text} {'(Valid)' if is_valid else '(Invalid)'}"
                        writer.writerow({"frame_number": frame_count, "license_text": plate_text, "confidence": max(conf for _, conf in plate_tracker[box_key] if _ == plate_text)})
                        logger.info("Video %s, Frame %d, Plate %d: %s", os.path.basename(video_path), frame_count, i+1, display_text)
                    else:
                        display_text = "Processing..."

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                out.write(frame)
                cv2.imshow("License Plate Detection and OCR", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        for _ in range(len(threads)):
            frame_queue.put(None)
        for t in threads:
            t.join()
        cap.release()
        out.release()
        logger.info("Output video saved to %s", output_path)
        logger.info("License plate details saved to %s", csv_file_path)

# Directory processing
def process_directory(input_dir, output_dir, csv_file_path):
    logger.info("Checking input directory: %s", input_dir)
    if not os.path.exists(input_dir):
        logger.error("Input directory %s does not exist", input_dir)
        exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]
    logger.info("Found videos: %s", video_files)

    if not video_files:
        logger.error("No video files found in %s", input_dir)
        exit(1)

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        process_video(video_path, output_dir, csv_file_path)

# Main execution
input_dir = "./videos"
output_dir = os.path.join(input_dir, "output")
csv_file_path = r"D:\Users\wwteo\Documents\ultralytics\ipi\license_plate_results.csv"
process_directory(input_dir, output_dir, csv_file_path)
cv2.destroyAllWindows()