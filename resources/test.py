from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/Users/fengbowen/Desktop/pythonProject/runs/detect/train3/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/Users/fengbowen/Desktop/pythonProject/resources/images_test", save=True, imgsz=640, conf=0.1)