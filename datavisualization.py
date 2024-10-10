import os
import random
import pydicom
import cv2
from PIL import Image

def visualize_data(csv_path, base_dir):
    """
    Load data and visualize a few random DICOM images from the dataset.
    """
    patient_labels, images = load_data(csv_path, base_dir)
    
    # Visualize a few random images
    num_images_to_show = 4
    random_indices = random.sample(range(len(images)), num_images_to_show)

    for i in random_indices:
        img = images[i]

        # Convert image to PIL format and show/save it
        pil_img = Image.fromarray(img)
        pil_img.save(f'sample_{i}.jpg')
        pil_img.show()

    return patient_labels, images

base_dir = 'rsna-2022-cervical-spine-fracture-detection/train_images'
csv_path = 'rsna-2022-cervical-spine-fracture-detection/train.csv'
    
visualize_data(csv_path, base_dir)
