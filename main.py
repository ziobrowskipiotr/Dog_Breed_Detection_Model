from prepare_dataset import prepare_dataset_with_dog_detection
from fine_tune import fine_tune_yolo_model
from split_dataset import split_dataset

if __name__ == "__main__":
    dataset_directory = 'Images' # Directory with our images selected by breeds
    dataset_directory_train = 'Images_train'  # Directory with dog images for training
    dataset_directory_val = 'Images_val'  # Directory with dog images for validation
    output_directory = '.'  # Directory where processed data will be saved
    
    # Move 20% of the data to the validation folder and 80% to train folder
    split_dataset(dataset_directory, dataset_directory_train, dataset_directory_val, val_ratio=0.2)
    # Prepare data in YOLO format (cropped dogs)
    prepare_dataset_with_dog_detection(dataset_directory_train, output_directory)
    prepare_dataset_with_dog_detection(dataset_directory_val, output_directory)

    # Fine-tune the YOLO model
    fine_tune_yolo_model()
