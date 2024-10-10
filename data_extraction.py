import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import boto3
import requests
import zipfile
from io import BytesIO

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'deeplearning-mlops'

def download_and_extract_data(zip_key):
    """Download and extract the dataset from S3."""
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket_name, 'Key': zip_key},
        ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
    )
    url_response = requests.get(url)
    url_response.raise_for_status()  # Raise an error for bad responses
    with zipfile.ZipFile(BytesIO(url_response.content)) as z:
        z.extractall('.')
    print("Data extraction successful.")

def load_data(csv_path, base_path, img_size=(224, 224)):
    """
    Load patient labels from CSV and DICOM images from the specified base path.
    Returns a tuple of (patient_labels, images).
    """
    # Load CSV
    data = pd.read_csv(csv_path)
    patient_labels = dict(zip(data['StudyInstanceUID'], data['patient_overall']))

    # Load DICOM images
    images = []
    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if os.path.isdir(patient_path):  # Check if it is a directory
            for dicom_file in os.listdir(patient_path):
                dicom_path = os.path.join(patient_path, dicom_file)
                if dicom_path.endswith('.dcm'):
                    try:
                        dicom = pydicom.dcmread(dicom_path, force=True)

                        if 'TransferSyntaxUID' not in dicom.file_meta:
                            dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

                        if not hasattr(dicom, 'PixelData'):
                            print(f"No pixel data for file: {dicom_path}")
                            continue

                        img = dicom.pixel_array
                        img = cv2.resize(img, img_size)
                        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel RGB
                        
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading DICOM file: {dicom_path}, {e}")
    
    return patient_labels, np.array(images)

# Main execution
zip_key = 'rsna-2022-cervical-spine-fracture-detection.zip'
download_and_extract_data(zip_key)

base_dir = 'rsna-2022-cervical-spine-fracture-detection/train_images'
csv_path = 'rsna-2022-cervical-spine-fracture-detection/train.csv'

load_data(csv_path, base_dir)
