from ultralytics import YOLO
from tkinter import filedialog, Tk

def convert_yolov11_to_onnx(model_path, input_shape=(1, 3, 640, 640), simplify=True):
    """
    Convert a trained Ultralytics YOLOv11 model to ONNX format.
    
    Args:
        model_path: Path to the trained YOLOv11 model (e.g., 'yolov11n.pt')
        input_shape: Tuple of input dimensions (batch, channels, height, width)
        simplify: Whether to simplify the ONNX model using onnx-simplifier
    """
    # Load the trained YOLOv11 model
    model = YOLO(model_path)
    
    # Generate ONNX file path by replacing .pt with .onnx
    onnx_file_path = model_path.rsplit('.pt', 1)[0] + '.onnx'
    
    # Export to ONNX
    model.export(
        format="onnx",
        imgsz=input_shape[2:],
        dynamic=True,
        simplify=simplify,
        opset=12
    )
    
    print(f"Model successfully converted to ONNX and saved at {onnx_file_path}")

def select_model_file():
    """
    Open a file dialog to select a YOLOv11 .pt model file.
    """
    # Initialize Tkinter (hide the root window)
    root = Tk()
    root.withdraw()
    
    # Open file dialog to select .pt file
    model_path = filedialog.askopenfilename(
        title="Select YOLOv11 Model File",
        filetypes=[("PyTorch Model Files", "*.pt")]
    )
    
    # Destroy the root window
    root.destroy()
    
    if model_path:
        print(f"Selected model: {model_path}")
        convert_yolov11_to_onnx(model_path)
    else:
        print("No file selected.")

# Run the file selection dialog
if __name__ == "__main__":
    select_model_file()