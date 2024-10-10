import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
import pydicom
import cv2
from PIL import Image
from datavisualization import visualize_data

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = Image.fromarray(image)  # Convert to PIL image
            image = self.transform(image)

        return image, label

def transform_data():
    # Load patient labels and images from visualize_data (which internally handles the paths)
    patient_labels, _ = visualize_data()  # We will not use images loaded here
    
    images = []
    labels = []

    base_dir = 'rsna-2022-cervical-spine-fracture-detection/train_images'  # Your base image directory

    # Iterate over each patient folder (StudyInstanceUID)
    for study_uid, label in patient_labels.items():
        study_folder = os.path.join(base_dir, study_uid)
        
        if os.path.isdir(study_folder):
            # For each DICOM file in the patient folder
            for dicom_file in os.listdir(study_folder):
                dicom_path = os.path.join(study_folder, dicom_file)
                
                if dicom_file.endswith('.dcm'):
                    try:
                        dicom = pydicom.dcmread(dicom_path)

                        if 'TransferSyntaxUID' not in dicom.file_meta:
                            dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

                        if not hasattr(dicom, 'PixelData'):
                            print(f"No pixel data for file: {dicom_path}")
                            continue

                        img = dicom.pixel_array
                        img = cv2.resize(img, (224, 224))  # Resize to the desired size
                        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel RGB

                        images.append(img)
                        labels.append(label)  # Append the same class label for all images in this folder
                    except Exception as e:
                        print(f"Error loading DICOM file: {dicom_path}, {e}")
        else:
            print(f"StudyInstanceUID folder not found: {study_uid}")
    
    # PyTorch transformations
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize the custom dataset
    dataset = CustomDataset(images=images, labels=labels, transform=data_transform)

    # Optionally, save the dataset using pickle
    with open('model_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    return dataset

# Example usage
transformed_dataset = transform_data()
