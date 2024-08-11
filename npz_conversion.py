import os
import numpy as np
from PIL import Image

def process_images_from_directory(directory_path, image_size=(64, 64)):
    image_list = []
    labels = []
    classes = sorted(os.listdir(directory_path))  # Sorted to maintain order
    for label, class_name in enumerate(classes):
        class_path = os.path.join(directory_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Convert to RGB if needed
                img = img.resize(image_size)  # Resize to a fixed size
                img_array = np.array(img)
                image_list.append(img_array)
                labels.append(label)
    
    images_array = np.array(image_list)
    labels_array = np.array(labels)
    return images_array, labels_array

def save_to_npz(train_dir, valid_dir, train_npz_path, valid_npz_path):
    train_images, train_labels = process_images_from_directory(train_dir)
    valid_images, valid_labels = process_images_from_directory(valid_dir)
    
    np.savez(train_npz_path, images=train_images, labels=train_labels)
    np.savez(valid_npz_path, images=valid_images, labels=valid_labels)

# Paths to your dataset directories and output .npz files
train_dir = r"C:\Users\panse\Downloads\PredRNN\PredRNN\asl_dataset"
valid_dir = r'C:\Users\panse\Downloads\PredRNN\PredRNN\asl_dataset_test'
train_npz_path = 'data/asl-train.npz'
valid_npz_path = 'data/asl-valid.npz'

save_to_npz(train_dir, valid_dir, train_npz_path, valid_npz_path)
