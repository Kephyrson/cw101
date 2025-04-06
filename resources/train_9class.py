from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Load a COCO-pretrained YOLO11n model, or define a new mdoel
    model = YOLO("C:/Users/kyy/Desktop/simple_code/cfg/models/yolo11_model_9class.yaml")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="C:/Users/kyy/Desktop/simple_code/cfg/datasets/yolo11_9class.yaml", 
                        epochs=500, imgsz=640)

