import cv2
import csv
import re
import numpy as np
import torch
import os
import subprocess
from ultralytics import YOLO
import logging
from fast_plate_ocr import LicensePlateRecognizer
import tkinter as tk
from tkinter import filedialog
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check OpenCV CUDA
cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
logger.info(f"OpenCV CUDA Available: {cuda_count} device(s)")
if cuda_count > 0:
    logger.info(f"CUDA Device ID: {cv2.cuda.getDevice()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device Name (via PyTorch): {torch.cuda.get_device_name(0)}")
# print(dir(cv2.cuda))  # List all attributes in cv2.cuda

# GPU performance test
img = np.zeros((2160, 3840, 3), dtype=np.uint8)
# Warm up CUDA
if cuda_count > 0:
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    _ = cv2.cuda.resize(gpu_img, (1920, 1080))
# CPU batch test
start = time.time()
for _ in range(10):
    cpu_img = cv2.resize(img, (1920, 1080))
cpu_time = (time.time() - start) / 10
logger.info(f"Average CPU resize time: {cpu_time:.4f} seconds")
# GPU batch test
if cuda_count > 0:
    gpu_img = cv2.cuda_GpuMat()
    start = time.time()
    for _ in range(10):
        gpu_img.upload(img)
        gpu_resized = cv2.cuda.resize(gpu_img, (1920, 1080))
        gpu_resized.download()
    gpu_time = (time.time() - start) / 10
    logger.info(f"Average GPU resize time: {gpu_time:.4f} seconds")

    
# Verify GPU availability for PyTorch (used by YOLO)
if torch.cuda.is_available():
    logger.info("PyTorch GPU Available: %s", torch.cuda.get_device_name(0))
else:
    logger.warning("PyTorch GPU unavailable, using CPU. Check CUDA/cuDNN installation.")

# Check FFmpeg NVDEC/NVENC support
try:
    result = subprocess.run(['ffmpeg', '-decoders'], capture_output=True, text=True)
    if 'hevc_cuvid' in result.stdout:
        logger.info("FFmpeg supports hevc_cuvid for hardware-accelerated decoding.")
    else:
        logger.warning("FFmpeg hevc_cuvid not available. Check FFmpeg build for NVDEC support.")
    result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
    if 'h264_nvenc' not in result.stdout:
        logger.error("FFmpeg h264_nvenc not available. Ensure FFmpeg is built with NVENC support.")
        exit(1)
except FileNotFoundError:
    logger.error("FFmpeg not found. Install FFmpeg with NVDEC/NVENC support.")
    exit(1)

# Load the YOLO model (ensure 4K processing)
model = YOLO(r".\best.pt")
model.to('cuda' if torch.cuda.is_available() else 'cpu')
print(model.device)  # Should print 'cuda:0'

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
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select Input Video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    logger.error("No video file selected.")
    exit(1)

# Generate output video and CSV paths based on input video name
video_dir = os.path.dirname(video_path) or '.'
video_base = os.path.splitext(os.path.basename(video_path))[0]
temp_video_path = os.path.join(video_dir, f"{video_base}_temp.mp4")
output_video_path = os.path.join(video_dir, f"{video_base}_processed.mp4")
csv_file_path = os.path.join(video_dir, f"{video_base}_results.csv")

# Regex pattern for valid license plate characters (alphanumeric, 5-7 characters)
plate_pattern = re.compile(r'^[A-Z0-9]{5,7}$')

# Open CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["license_text", "yolo_confidence", "ocr_confidence"])
    writer.writeheader()

    # Capture video with FFmpeg backend for NVDEC decoding
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open video file: %s", video_path)
        exit(1)

    # Get video properties (expecting 4K: 3840x2160)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"Input video resolution: {frame_width}x{frame_height}, FPS: {fps}")
    if frame_width != 3840 or frame_height != 2160:
        logger.warning("Input video is not 4K (3840x2160). Detected: %dx%d", frame_width, frame_height)

    # Initialize VideoWriter for temporary output (same resolution as original)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        logger.error("Failed to initialize VideoWriter for: %s", temp_video_path)
        cap.release()
        exit(1)

    processed_plates = set()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Verify frame resolution
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            logger.warning(f"Frame {frame_count} resolution mismatch: expected {frame_width}x{frame_height}, got {frame.shape[1]}x{frame.shape[0]}")

        # Step 1: Detect license plates using YOLO at 4K
        try:
            results = model.track(frame, persist=True, classes=[0], conf=0.6, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.error(f"Error during YOLO tracking on frame {frame_count}: {e}")
            continue

        if len(results) > 0 and results[0].boxes is not None:
            # Get the detected boxes and confidence scores
            boxes = results[0].boxes.xyxy.cpu().numpy()
            yolo_confidences = results[0].boxes.conf.cpu().numpy()

            # Step 2: Process each detected object
            for box_idx, (box, yolo_conf) in enumerate(zip(boxes, yolo_confidences)):
                x_min, y_min, x_max, y_max = map(int, box)

                # Crop the license plate area (no resizing or preprocessing)
                if cuda_count > 0:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    # Create a submatrix (ROI) on GPU
                    gpu_license_plate = cv2.cuda_GpuMat(gpu_frame, (x_min, y_min, x_max - x_min, y_max - y_min))
                    license_plate = gpu_license_plate.download()  # Download for OCR
                else:
                    license_plate = frame[y_min:y_max, x_min:x_max]

                # Run OCR using fast-plate-ocr
                try:
                    ocr_result = ocr.run(source=license_plate, return_confidence=True)
                    print(f"Raw OCR output for box {box_idx + 1}: {ocr_result}")
                    if ocr_result and len(ocr_result) == 2 and len(ocr_result[0]) > 0:
                        for plate_idx, plate_text in enumerate(ocr_result[0]):
                            # Ensure confidence array is a NumPy array
                            ocr_conf_array = ocr_result[1][plate_idx] if len(ocr_result[1]) > plate_idx and isinstance(ocr_result[1][plate_idx], np.ndarray) else np.array([])
                            # Clean text: remove all underscores
                            clean_text = plate_text.strip().upper().replace('_', '') if plate_text else ""
                            # Filter for valid license plate format
                            if clean_text and not plate_pattern.match(clean_text):
                                clean_text = ""
                                ocr_conf_array = np.array([])
                            # Convert confidence array to list for CSV storage
                            ocr_conf_list = [float(conf) for conf in ocr_conf_array] if ocr_conf_array.size > 0 else []
                            # Compute average confidence for video annotation
                            ocr_conf_avg = float(np.mean(ocr_conf_array)) if ocr_conf_array.size > 0 else 0.0

                            # Check if the license plate is valid and not already processed
                            if clean_text and clean_text not in processed_plates:
                                processed_plates.add(clean_text)
                                writer.writerow({
                                    "license_text": clean_text,
                                    "yolo_confidence": f"{yolo_conf:.2f}",
                                    "ocr_confidence": str(ocr_conf_list)
                                })
                                print(f"New license plate detected: {clean_text} (YOLO Conf: {yolo_conf:.2f}, OCR Conf: {ocr_conf_avg:.2f})")

                            # Draw detection results on the frame
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            label = f"{clean_text} (YOLO: {yolo_conf:.2f}, OCR: {ocr_conf_avg:.2f})"
                            cv2.putText(frame, label, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    else:
                        clean_text, ocr_conf_list, ocr_conf_avg = "", [], 0.0
                        logger.warning(f"No valid OCR result for box {box_idx + 1} in frame {frame_count}")

                except Exception as e:
                    logger.error(f"Error processing OCR for box {box_idx + 1} on frame {frame_count}: {e}")
                    continue

        # Write the annotated frame to the temporary output video
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

# Compress the temporary video using FFmpeg with NVDEC (hevc_cuvid) and NVENC
try:
    ffmpeg_cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',     # Enable NVDEC (hevc_cuvid or h264_cuvid)
        '-i', temp_video_path,
        '-c:v', 'h264_nvenc',   # Use NVIDIA NVENC H.264 encoder
        '-rc', 'vbr',           # Variable bitrate mode
        '-cq', '23',            # Constant quality
        '-preset', 'p7',        # Highest quality preset
        '-c:a', 'aac',          # Audio codec
        '-vf', f'scale=1920:1080',  # Ensure 4K resolution
        '-y',                   # Overwrite output
        output_video_path
    ]
    # Optional: Use hevc_nvenc for better compression (uncomment if desired)
    # ffmpeg_cmd = [
    #     'ffmpeg',
    #     '-hwaccel', 'cuda',
    #     '-i', temp_video_path,
    #     '-c:v', 'hevc_nvenc',
    #     '-rc', 'vbr',
    #     '-cq', '23',
    #     '-preset', 'p7',
    #     '-c:a', 'aac',
    #     '-vf', f'scale=3840:2160',
    #     '-y',
    #     output_video_path
    # ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Video compressed successfully with NVDEC/NVENC: {output_video_path}")
        os.remove(temp_video_path)
        logger.info(f"Temporary video deleted: {temp_video_path}")
    else:
        logger.error(f"FFmpeg compression failed: {result.stderr}")
        output_video_path = temp_video_path
except FileNotFoundError:
    logger.error("FFmpeg not found. Ensure FFmpeg is installed with NVDEC/NVENC support.")
    output_video_path = temp_video_path
except Exception as e:
    logger.error(f"Error during FFmpeg compression: {e}")
    output_video_path = temp_video_path

print(f"License plate details saved to {csv_file_path}")
print(f"Annotated video saved to {output_video_path}")