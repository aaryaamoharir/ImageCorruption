import os 
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corruptions as corruptions
import Corruption_transform as Corruption_transform
from Corruption_transform import uncorrelated_corruption_transform
print("hi")
data_path = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /minorityML/data"

def load_data(data_path):
    image_paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")  # Convert to RGB in case some are grayscale or CMYK
            images.append((img, path)) 
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    print(f"Loaded {len(images)} images.")
    
    return images

def corrupt_images(images):
    b = 0
    print("hi in corrupt_images")
    output_dir = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /minorityML/data/corrupted_images"
    os.makedirs(output_dir, exist_ok=True)
    corruption = uncorrelated_corruption_transform(level=3, type='all')
    print(f"Loaded {len(images)} images for corrupted images.")
    for i, (img, path) in enumerate(images):
        try:
            print(b)
            corrupted_img, corruption_name = corruption(img, modality="RGB")
            base = os.path.basename(path)
            name, ext = os.path.splitext(base)
            new_filename = f"{name}_{corruption_name}_corrupted{b}{ext}"
            save_path = os.path.join(output_dir, new_filename)
            corrupted_img.save(save_path)
            # uncomment for progress logging
            # print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Failed to corrupt/save image {path}: {e}")
        b = b + 1


if __name__ == "__main__":
    images = load_data(data_path)
    print(f"Loaded {len(images)} images.")
    print(type(images[0]), images[0])
    corrupt_images(images)
    print("done corrupting images")

   
