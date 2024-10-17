# Dog_Breed_Detection_Model

## Project Overview

This project focuses on fine-tuning the YOLOv8n model for detecting and classifying 120 different dog breeds using a custom dataset. The model is trained to recognize dogs and locate them in images, while providing the corresponding breed. The project includes steps to prepare the dataset, split it for training and validation, and detect dogs using the fine-tuned model.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Summary](#summary)
- [References](#references)

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:ziobrowskipiotr/Dog_Breed_Detection_Model.git
   cd Dog_Breed_Detection_Model
   ```

2. In the [Data release](https://github.com/ziobrowskipiotr/Dog_Breed_Detection_Model/releases/tag/Data), you can find the following files:

- **Images.zip**: Contains over 16,000 labeled images of dog breeds, which are used to train the YOLOv8 model.
- **example_prep_data.zip**: Contains sample pre-processed data, which showcases how the images were prepared for the model.

### How to Use

1. Download the `data.zip` from the [Data release](https://github.com/ziobrowskipiotr/Dog_Breed_Detection_Model/releases/tag/Data).
2. Extract the `data.zip` file, which includes:
   - `Images.zip` – the full dataset of labeled dog breed images.
   - `example_prep_data.zip` – example data prepared for training.
   - Make sure to extract `Images.zip` to your desired working directory before starting the training process.
3. **Download the YOLOv8 pre-trained model: YOLOv8 can be used with a pre-trained model on the COCO dataset, where class 16 corresponds to dogs:**
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

## Summary

The YOLOv8 model was trained for 382 epochs with a final performance of:

- Precision (P): 0.65
- Recall (R): 0.674
- mAP@50: 0.704
- mAP@50-95: 0.614

While the model achieved reasonable performance metrics, the primary limitation was the dataset size. With only 4,072 images and 3,295 instances, the data proved insufficient to fully differentiate between 120 dog breeds. The limited dataset size likely prevented further improvement beyond the observed results.

Additionally, the model was stopped early due to EarlyStopping criteria, where no significant improvement was observed after 50 epochs. The best model was saved at epoch 334 with the highest mAP@50 of 0.704.

Due to technical constraints, it was not possible to expand the project further. However, the structure of the project has been designed in a way that allows for easy continuation by other contributors. With more training data and additional resources, the model could be fine-tuned further to improve accuracy and robustness in detecting and classifying dog breeds.


## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [COCO Dataset Classes](https://cocodataset.org/#home)
