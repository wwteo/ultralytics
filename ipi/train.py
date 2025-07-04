from ultralytics import YOLO
if __name__ == '__main__':
    # Load a pretrained YOLOv8 model
    # model = YOLO("yolov8n.pt")  # Load a pretrained model
    # model = YOLO("yolov8n-seg.pt")  # Load a pretrained segmentation model
    # model = YOLO("yolov8n-pose.pt")  # Load a pretrained pose estimation model

    # Train the model
    # 1. Specify the path to your dataset YAML file
    # 2. Set the number of epochs, image size, batch size, device, learning rate, and augmentation options
    model = YOLO("../anpr-demo-model.pt")
    model.train(data="../data.yaml", epochs=50, imgsz=1920, batch=4, device=0, lr0=0.001, augment=True)
