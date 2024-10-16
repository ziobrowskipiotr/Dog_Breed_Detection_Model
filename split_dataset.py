import os
import shutil
import random

def split_dataset(dataset_dir, output_train_dir, output_val_dir, val_ratio=0.2):
    # Create directories if they do not exist
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    for breed in sorted(os.listdir(dataset_dir)):
        breed_dir = os.path.join(dataset_dir, breed)

        if os.path.isdir(breed_dir):  # Check if it's a folder
            # Create target folders for training and validation
            train_breed_dir = os.path.join(output_train_dir, breed)
            val_breed_dir = os.path.join(output_val_dir, breed)
            os.makedirs(train_breed_dir, exist_ok=True)
            os.makedirs(val_breed_dir, exist_ok=True)

            # Get all images from the breed folder
            images = [img for img in os.listdir(breed_dir) if img.endswith((".jpg", ".png"))]

            # Shuffle images and split into training and validation sets
            random.shuffle(images)
            val_size = int(len(images) * val_ratio)
            val_images = images[:val_size]
            train_images = images[val_size:]

            # Move images to training and validation folders
            for img in train_images:
                shutil.copy(os.path.join(breed_dir, img), os.path.join(train_breed_dir, img))

            for img in val_images:
                shutil.copy(os.path.join(breed_dir, img), os.path.join(val_breed_dir, img))

    print("Splitting into training and validation sets completed.")

if __name__ == "__main__":
    dataset_directory = 'Images'  # Directory with original data
    output_train_directory = 'Images_train'  # Directory for training data
    output_val_directory = 'Images_val'  # Directory for validation data

    # Move 20% of the data to the validation folder
    split_dataset(dataset_directory, output_train_directory, output_val_directory, val_ratio=0.2)
