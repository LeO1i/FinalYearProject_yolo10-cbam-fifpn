import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

train_path = str(DATASET_PATH / "train")
val_path = str(DATASET_PATH / "val")

classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to load images and labels
def load_data(data_path):
    images = []
    labels = []
    for class_label in classes:
        class_path = os.path.join(data_path, class_label, 'images')
        label_path = os.path.join(data_path, class_label, 'labels')
        for img_file in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label_file = img_file.replace('.jpg', '.txt')
            label_file_path = os.path.join(label_path, label_file)
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as file2:
                    label_data = file2.readline().strip().split()
                    if len(label_data) > 0:
                        images.append(img)
                        labels.append(label_data)
                    else:
                        print(f"Label file {label_file_path} is empty, skipping this image.")
            else:
                print(f"Label file {label_file_path} not found, skipping this image.")
    return images, labels

# Load training and validation data


def preprocess_images(images):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (640, 640))
        processed_images.append(img_resized)
    return np.array(processed_images)

dataset_yaml = {
    'path': str(DATASET_PATH),
    'train': 'train',
    'val': 'val',
    'names':  classes
}
dataset_yaml_path = PROJECT_ROOT / 'dataset.yaml'
with open(dataset_yaml_path, 'w') as file:
    yaml.dump(dataset_yaml, file)

if __name__ == '__main__':
    #print(torch.cuda.is_available())
    train_images, train_labels = load_data(train_path)
    val_images, val_labels = load_data(val_path)
    train_images = preprocess_images(train_images)
    val_images = preprocess_images(val_images)

    # Load YOLOv10 model with CBAM
    model_config = Path(__file__).parent / "yolov10n_CBAM.yaml"
    model = YOLO(str(model_config))

    # Train the model
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    result = model.train(data=str(dataset_yaml_path),
                         epochs=10,  # Increase training to 1000
                         imgsz=640,     # Image size
                         lr0=0.001,  # Initial learning rate
                         lrf=0.2,  # Final learning rate multiplier
                         mosaic=True,  # Enable mosaic augmentation
                         mixup=True,  # Enable mixup augmentation
                         hsv_h=0.015,  # Adjust hue
                         hsv_s=0.7,  # Adjust saturation
                         hsv_v=0.4,  # Adjust value (brightness)
                         batch=16,  # Batch size (adjust based on GPU capacity)
                         workers=4,  # Number of data loader workers
                         deterministic = False,
                         )

    # Save model to Trained_model directory
    output_path = PROJECT_ROOT / 'Trained_model' / 'YOLOv10CM_FYPtrained_new.pt'
    model.save(str(output_path))
    print(f"\nModel saved to: {output_path}")























