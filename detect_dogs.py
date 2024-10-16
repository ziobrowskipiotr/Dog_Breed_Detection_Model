from ultralytics import YOLO
from PIL import Image

def detect_dogs(image_path):
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is a pretrained model file

    # Perform object detection on the input image
    results = model(image_path)

    # Process detection results to find dogs (class 16 in COCO dataset)
    for result in results:
        for box in result.boxes:  # Iterate over detected bounding boxes
            if int(box.cls[0]) == 16:  # Check if the detected object is a dog (class 16)
                # Extract bounding box coordinates (top-left and bottom-right corners)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Open the input image to get its dimensions
                img = Image.open(image_path)
                img_width, img_height = img.size

                # Convert bounding box coordinates to YOLO format (normalized center and size)
                x_center = (x1 + x2) / 2 / img_width  # Normalize the x-center by image width
                y_center = (y1 + y2) / 2 / img_height  # Normalize the y-center by image height
                width = (x2 - x1) / img_width  # Normalize the width of the bounding box
                height = (y2 - y1) / img_height  # Normalize the height of the bounding box

                # Return the normalized coordinates in YOLO format
                return [x_center, y_center, width, height]

    return None  # Return None if no dog is detected
