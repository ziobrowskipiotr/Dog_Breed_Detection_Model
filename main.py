from prepare_dataset import prepare_dataset_with_dog_detection
from fine_tune import fine_tune_yolo_model

if __name__ == "__main__":
    dataset_directory = 'Images_train'  # Directory with dog images for training
    dataset_directory1 = 'Images_val'  # Directory with dog images for validation
    output_directory = '.'  # Directory where processed data will be saved

    # Prepare data in YOLO format (cropped dogs)
    prepare_dataset_with_dog_detection(dataset_directory, output_directory)
    prepare_dataset_with_dog_detection(dataset_directory1, output_directory)

    # Fine-tune the YOLO model
    fine_tune_yolo_model()
