import os
from pathlib import Path
from detect_dogs import detect_dogs

def prepare_dataset_with_dog_detection(dataset_dir, output_dir):
    classes = []

    for idx, breed in enumerate(sorted(os.listdir(dataset_dir))):
        labels_dir = os.path.join(dataset_dir, breed)

        if os.path.isdir(os.path.join(dataset_dir, breed)):  # Check if it's a directory
            classes.append(breed)
            breed_dir = os.path.join(dataset_dir, breed)

            # Process images in the breed directory
            for img_file in os.listdir(breed_dir):
                if img_file.endswith((".jpg", ".png")):
                    img_path = os.path.join(breed_dir, img_file)

                    # Detect the dog
                    bbox = detect_dogs(img_path)
                    if bbox is not None:
                        # Create a label file in YOLO format only if a dog is detected and saved
                        label_file = f"{Path(img_file).stem}.txt"
                        label_path = os.path.join(labels_dir, label_file)

                        with open(label_path, "w") as f:
                            f.write(f"{idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    # Save the file with class names
    with open(os.path.join(output_dir, "classes.names"), "w") as f:
        for breed in classes:
            f.write(f"{breed}\n")
