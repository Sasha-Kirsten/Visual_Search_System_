import shutil
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils


dataset = LightlyDataset("\Users\Besitzer\Desktop\Image_Dataset", transform=None)

dataloader = torch.utils.DataLoader(
    dataset, batch_size=256, shuffle=True,
    drop_last=True, num_workers=8
)




# def get_all_image_paths(root_dir):
#     """
#     Recursively gets all image file paths from a root directory and its subfolders.
#     """
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.JPG', '.JPEG', '.PNG'}
#     image_paths = []
    
#     for current_dir, subdirs, files in os.walk(root_dir):
#         for file in files:
#             # Get the file extension
#             _, ext = os.path.splitext(file)
#             if ext in image_extensions:
#                 full_path = os.path.join(current_dir, file)
#                 image_paths.append(full_path)
                
#     return image_paths

# def copy_files(file_paths, source_dir, dest_dir):
#     """
#     Copies files from a list of full paths to a destination directory,
#     preserving the subfolder structure relative to the source_dir.
#     """
#     os.makedirs(dest_dir, exist_ok=True)
    
#     for src_path in file_paths:
#         # Get the relative path from the source directory
#         rel_path = os.path.relpath(src_path, source_dir)
#         dest_path = os.path.join(dest_dir, rel_path)
        
#         # Create the necessary subdirectories in the destination
#         os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#         shutil.copy2(src_path, dest_path)
#         print(f"Copied: {rel_path}")

# # --- Main Execution ---
# source_dir = r"C:\Users\Besitzer\Desktop\Dataset"

# # 1. Get ALL image paths recursively
# print(f"Searching for images in: {source_dir}")
# all_image_paths = get_all_image_paths(source_dir)

# print(f"Found {len(all_image_paths)} images.")

# if len(all_image_paths) == 0:
#     print("Error: No images found. Please check the following:")
#     print(f"1. Does the path '{source_dir}' exist?")
#     print(f"2. Are there any image files in this directory or its subfolders?")
#     print(f"3. Common image extensions looked for: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
#     # Let's also list what *is* in the top directory for debugging
#     if os.path.exists(source_dir):
#         print("\nContents of the directory:")
#         for item in os.listdir(source_dir):
#             print(f"  - {item}")
#     exit() # Stop the script if no images are found

# # 2. Split the list of paths (not just filenames!)
# train_list, val_list = train_test_split(all_image_paths, test_size=0.1, random_state=42)

# print(f"Training set: {len(train_list)} images")
# print(f"Validation set: {len(val_list)} images")

# # 3. Define target directories
# train_dir = os.path.join(source_dir, 'train')
# val_dir = os.path.join(source_dir, 'val')

# # 4. Copy the files to the new train/val folders
# print("\nCopying training images...")
# copy_files(train_list, source_dir, train_dir)

# print("\nCopying validation images...")
# copy_files(val_list, source_dir, val_dir)

# print("\nDone! Dataset successfully split into train and validation sets.")
# print(f"Training images: {train_dir}")
# print(f"Validation images: {val_dir}")




from pathlib import Path

def create_train_val_split(source_dir, output_base_dir, val_size=0.1, random_seed=42):
    """
    Splits images from a source directory into train and validation sets.
    
    Args:
        source_dir (str): Path to the folder containing all images.
        output_base_dir (str): Path where the 'train' and 'val' folders will be created.
        val_size (float): Proportion of data to use for validation (e.g., 0.1 for 10%).
        random_seed (int): Seed for reproducible splits.
    """
    source_path = Path(source_dir)
    all_images = [
        f for f in source_path.iterdir() 
        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    ]
    
    print(f"Found {len(all_images)} total images.")
    
    train_list, val_list = train_test_split(
        all_images, 
        test_size=val_size, 
        random_state=random_seed
    )
    
    train_dir = Path(output_base_dir) / "train"
    val_dir = Path(output_base_dir) / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_files(file_list, destination):
        for file_path in file_list:
            shutil.copy2(file_path, destination / file_path.name)
    
    print("Copying training images...")
    copy_files(train_list, train_dir)
    print("Copying validation images...")
    copy_files(val_list, val_dir)
    
    print("Split complete!")
    print(f"Training images: {len(train_list)}")
    print(f"Validation images: {len(val_list)}")
    print(f"Training folder: {train_dir}")
    print(f"Validation folder: {val_dir}")

if __name__ == '__main__':
    SOURCE_DIR = r"C:\Users\Besitzer\Desktop\Dataset"
    OUTPUT_DIR = r"C:\Users\Besitzer\Desktop\Image_Dataset_Split"
    
    create_train_val_split(SOURCE_DIR, OUTPUT_DIR, val_size=0.1)
