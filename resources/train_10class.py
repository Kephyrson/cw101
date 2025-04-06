from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Load a COCO-pretrained YOLO11n model, or define a new mdoel
    model = YOLO("/Users/fengbowen/Desktop/pythonProject/resources/cfg/models/yolo11_model_10class.yaml").load("best.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="/Users/fengbowen/Desktop/pythonProject/resources/cfg/datasets/yolo11_10class.yaml", 
                        epochs=3, imgsz=640)

