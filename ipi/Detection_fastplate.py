import cv2
import csv
import re
import numpy as np
import torch
import os
from ultralytics import YOLO
import logging
from fast_plate_ocr import LicensePlateRecognizer
import tkinter as tk
from tkinter import filedialog

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify GPU availability for PyTorch (used by YOLO)
if torch.cuda.is_available():
    logger.info("PyTorch GPU Available: %s", torch.cuda.get_device_name(0))
else:
    logger.warning("PyTorch GPU unavailable, using CPU. Check CUDA/cuDNN installation.")

# Load the YOLO model (automatically uses GPU if available)
model = YOLO(r".\best.pt")
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize fast-plate-ocr with European model and GPU support
try:
    ocr = LicensePlateRecognizer(hub_ocr_model="cct-xs-v1-global-model", device='cuda')
    logger.info("FastPlateOCR initialized with CUDAExecutionProvider for European/UK plates.")
except Exception as e:
    logger.error("Failed to initialize FastPlateOCR with GPU: %s", e)
    logger.info("Falling back to CPU provider.")
    try:
        ocr = LicensePlateRecognizer("cct-xs-v1-global-model")
        logger.info("FastPlateOCR initialized with CPUExecutionProvider.")
    except Exception as e:
        logger.error("Failed to initialize FastPlateOCR on CPU: %s", e)
        exit(1)

# Open file dialog to select input video
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
video_path = filedialog.askopenfilename(
    title="Select Input Video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    logger.error("No video file selected.")
    exit(1)

# Generate output video and CSV paths based on input video name
video_dir = os.path.dirname(video_path) or '.'  # Use current directory if no dirname
video_base = os.path.splitext(os.path.basename(video_path))[0]
output_video_path = os.path.join(video_dir, f"{video_base}_processed.mp4")
csv_file_path = os.path.join(video_dir, f"{video_base}_results.csv")

# Regex pattern for valid license plate characters (alphanumeric, 5-7 characters)
plate_pattern = re.compile(r'^[A-Z0-9]{5,7}$')

# Open CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["license_text", "yolo_confidence", "ocr_confidence"])
    writer.writeheader()

    # Capture video (read-only, original video is preserved)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file: %s", video_path)
        exit(1)

    processed_plates = set()
    frame_count = 0

    # Get video properties (maintain original resolution: 1088x1920)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter for annotated output (same resolution as original)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        logger.error("Failed to initialize VideoWriter for: %s", output_video_path)
        cap.release()
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Step 1: Detect license plates using YOLO
        try:
            results = model.track(frame, persist=True, classes=[0], conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.error(f"Error during YOLO tracking on frame {frame_count}: {e}")
            continue

        if len(results) > 0 and results[0].boxes is not None:
            # Get the detected boxes and confidence scores
            boxes = results[0].boxes.xyxy.cpu().numpy()
            yolo_confidences = results[0].boxes.conf.cpu().numpy()

            # Step 2: Process each detected object
            for box, yolo_conf in zip(boxes, yolo_confidences):
                x_min, y_min, x_max, y_max = map(int, box)

                # Crop the license plate area (no resizing or preprocessing)
                license_plate = frame[y_min:y_max, x_min:x_max]

                # Run OCR using fast-plate-ocr
                try:
                    ocr_result = ocr.run(source=license_plate, return_confidence=True)
                    print(f"Raw OCR output: {ocr_result}")
                    if ocr_result and len(ocr_result) == 2 and len(ocr_result[0]) > 0:
                        plate_text = ocr_result[0][0]  # Extract first plate text
                        ocr_conf_array = ocr_result[1][0]  # Extract confidence array
                        # Clean text: remove unwanted characters and convert to uppercase
                        clean_text = plate_text.strip().upper().replace('__', '') if plate_text else ""
                        # Filter for valid license plate format
                        if clean_text and not plate_pattern.match(clean_text):
                            clean_text = ""
                            ocr_conf_array = []
                        # Convert confidence array to list for CSV storage
                        ocr_conf_list = [float(conf) for conf in ocr_conf_array] if ocr_conf_array.size > 0 else []
                        # Compute average confidence for video annotation
                        ocr_conf_avg = float(np.mean(ocr_conf_array)) if ocr_conf_array.size > 0 else 0.0
                    else:
                        clean_text, ocr_conf_list, ocr_conf_avg = "", [], 0.0

                    # Check if the license plate is valid and not already processed
                    if clean_text and clean_text not in processed_plates:
                        processed_plates.add(clean_text)
                        writer.writerow({
                            "license_text": clean_text,
                            "yolo_confidence": f"{yolo_conf:.2f}",
                            "ocr_confidence": str(ocr_conf_list)  # Store as stringified list
                        })
                        print(f"New license plate detected: {clean_text} (YOLO Conf: {yolo_conf:.2f}, OCR Conf: {ocr_conf_avg:.2f})")

                    # Draw detection results on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{clean_text} (YOLO: {yolo_conf:.2f}, OCR: {ocr_conf_avg:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                except Exception as e:
                    logger.error(f"Error processing OCR on frame {frame_count}: {e}")
                    continue

        # Write the annotated frame to the output video
        out.write(frame)

        # Optionally display the frame
        cv2.imshow("License Plate Detection and OCR", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User interrupted processing with 'q' key.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"License plate details saved to {csv_file_path}")
print(f"Annotated video saved to {output_video_path}")