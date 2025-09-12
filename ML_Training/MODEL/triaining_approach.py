# Step 1: Organize Your Image Files
# You don't need any labels or complex folder structures. The simplest way is to put all your fruit images into a single folder.

# text
# my_fruit_dataset/
# └── all_images/
#     ├── apple_001.jpg
#     ├── apple_002.jpg
#     ├── banana_001.jpg
#     ├── banana_002.jpg
#     └── ... (all other images)


# Step 2: Define the Augmentations (The Key to Self-Supervision)
# The lightly library provides built-in transform classes that handle the two-view augmentation needed for contrastive loss (like NT-Xent). We'll use SimCLRTransform which is a standard set of augmentations for this purpose.

# python
# from lightly.transforms import SimCLRTransform

# # Define the transform
# # This will create two random augmented versions of each input image
# transform = SimCLRTransform(
#     input_size=224,        # Output size of the image
#     Gaussian_blur=0.5,     # Probability of applying Gaussian blur
# )


# Step 3: Create a PyTorch Dataset Class
# You need to create a dataset that loads images from your folder and applies the transform. Since there are no labels, this is very straightforward.

# python
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# class FruitDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
        
#         # Get list of all image file paths
#         self.image_paths = []
#         for file_name in os.listdir(root_dir):
#             if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 self.image_paths.append(os.path.join(root_dir, file_name))
                
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')  # Ensure it's RGB
        
#         # Apply transform - which returns two views (x0, x1)
#         if self.transform:
#             x0 = self.transform(image)
#             x1 = self.transform(image)
#             return x0, x1  # Return the two augmented views
        
#         return image