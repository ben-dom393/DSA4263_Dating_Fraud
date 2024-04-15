# Required libraries
import os
import glob
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data
real_path = 'data/raw/face_real'
fake_path = 'data/raw/face_fake/1m_faces_00'

images_df = {
    "image_path": [],
    "label": []
}

# Process real images
for subfolder in os.listdir(real_path):
    full_path = os.path.join(real_path, subfolder)
    if os.path.isdir(full_path):  # Check if it is a directory
        for file_extension in ["*.jpg", "*.png"]:
            for img in glob.glob(os.path.join(full_path, file_extension)):
                images_df["image_path"].append(img)
                images_df["label"].append('real')

# Process fake images
count = 0
for file_extension in ["*.jpg", "*.png"]:
    for img in glob.glob(os.path.join(fake_path, file_extension)):
        images_df["image_path"].append(img)
        images_df["label"].append('fake')

images_df = pd.DataFrame(images_df)
shape = images_df.shape
print(f"Shape of the DataFrame: {shape}")
print("First few rows of the DataFrame:")
print(images_df.head())
print("Summary of the DataFrame:")
print(images_df.info())

def resize_image(image: Image.Image, size: tuple[int, int] = (128, 128)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).to(device)

def normalize_pixels(image_tensor: torch.Tensor) -> torch.Tensor:
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform(image_tensor)

def convert_to_grayscale(image_tensor: torch.Tensor) -> torch.Tensor:
    transform = transforms.Grayscale(num_output_channels=3)
    return transform(image_tensor)

def preprocess_pipeline(image_path: str, size: tuple[int, int] = (128, 128)) -> torch.Tensor:
    try:
        image = Image.open(image_path)
        image_tensor = resize_image(image, size=size)
        image_tensor = normalize_pixels(image_tensor)
        image_tensor = convert_to_grayscale(image_tensor)
        # Additional processing steps would go here
        return image_tensor
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def save_preprocessed_image(image_tensor, save_path, image_name):
    if image_tensor is not None:
        os.makedirs(save_path, exist_ok=True)
        save_full_path = os.path.join(save_path, image_name)
        # Convert tensor to PIL Image to save
        image_tensor = image_tensor.cpu().clone()  # Move to CPU and clone tensor
        image = transforms.ToPILImage()(image_tensor)
        image.save(save_full_path)
    else:
        print(f"Skipping saving for {image_name} as preprocessing failed")

save_directory_real = 'data/interim/face_processed/real'
save_directory_fake = 'data/interim/face_processed/fake'

for index, row in images_df.iterrows():
    preprocessed_image = preprocess_pipeline(
        image_path=row['image_path'],
        size=(128, 128),
    )
    image_name = os.path.basename(row['image_path'])
    if preprocessed_image is not None:
        if row['label'] == 'real':
            save_preprocessed_image(preprocessed_image, save_directory_real, image_name)
        else:
            save_preprocessed_image(preprocessed_image, save_directory_fake, image_name)
            print(f"Processed and saved {image_name}")
    else:
        print(f"Failed to process {image_name}")
