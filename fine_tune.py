from ultralytics import YOLO

def fine_tune_yolo_model():
    # Load the YOLO model
    model = YOLO('yolov8n.pt')  # YOLO pretrained on COCO (where class 16 is a dog)

    # Train the model on your data (containing various dog breeds)
    model.train(data='dataset.yaml', epochs=500, imgsz=320, freeze=10)  # Adjust the number of epochs and other parameters as needed

    # Save the fine-tuned model
    model.save('my_yolov8n.pt')
