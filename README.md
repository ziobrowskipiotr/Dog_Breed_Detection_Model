# Dog_Breed_Detection_Model

## Project Overview

This project focuses on fine-tuning the YOLOv8n model for detecting and classifying 120 different dog breeds using a custom dataset. The model is trained to recognize dogs and locate them in images, while providing the corresponding breed. The project includes steps to prepare the dataset, split it for training and validation, and detect dogs using the fine-tuned model.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [References](#references)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/dog-breed-detection-yolov8.git
   cd Dog_Breed_Detection_Model
   ```
2. **Download the YOLOv8 pre-trained model: YOLOv8 can be used with a pre-trained model on the COCO dataset, where class 16 corresponds to dogs:**
   ```bash
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   ```
## Dataset Preparation

Before training, we need to prepare and split the dataset.

### Split Dataset:

1. The `split_dataset.py` script splits the dataset into training and validation sets, with 80% of the data allocated to training and 20% to validation.

   Example usage:

   ```bash
   python split_dataset.py
   ```

### Prepare Dataset:

2. The script `prepare_dataset.py` processes images from the dataset and detects dogs, converting their bounding boxes to YOLO format.
   - It detects dogs in each image and creates corresponding YOLO label files in `.txt` format.

   Example usage:

   ```bash
   python prepare_dataset.py
   ```
## Detecting Dog Breeds

Function prepare_dataset use the `detect_dogs.py` script to detect dog breeds in images using model YOLOv8n.

```python
from ultralytics import YOLO
from PIL import Image

def detect_dogs(image_path):
    model = YOLO('my_yolov8n.pt')  # Load the fine-tuned model
    results = model(image_path)  # Perform detection
    # Further processing to extract and print breed details
```

## Training the Model

Fine-tune the YOLOv8 model using your custom dataset by running the `fine_tune_yolo_model()` function.

```python
  from ultralytics import YOLO

  def fine_tune_yolo_model():
      # Load the YOLO model
      model = YOLO('yolov8n.pt')  # Pre-trained model

      # Train the model on your dataset
      model.train(data='dataset.yaml', epochs=500, imgsz=320, freeze=10)

      # Save the fine-tuned model
      model.save('my_yolov8n.pt')
```
To run the fine-tuning:
```bash
  python fine_tune_model.py
```

## Project Structure
```bash
  ├── dataset.yaml                # Configuration file for dataset
  ├── detect_dogs.py              # Script for detecting dog breeds in images
  ├── fine_tune_model.py          # Script to fine-tune YOLOv8 model
  ├── prepare_dataset.py          # Script to prepare dataset for YOLOv8
  ├── split_dataset.py            # Script to split dataset into train and validation sets
  ├── README.md                   # Project documentation
```

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [COCO Dataset Classes](https://cocodataset.org/#home)
