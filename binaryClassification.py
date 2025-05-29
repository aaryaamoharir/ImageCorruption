import os
from tqdm import tqdm
import shutil
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
#from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # List all image file paths recursively (or in one folder)
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(dirpath, fname))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Assign label based on whether 'corrupted' is in the path
        label = 1 if 'corrupted' in img_path else 0
        if self.transform:
            image = self.transform(image)
        return image, label
        
def convert_to_one_directory(file_path):
    print("hi in model")
    target_path = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /minorityML/combined/"
    os.makedirs(target_path, exist_ok=True)
    # load images from the specified directory
    image_paths = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))

    images = []
    count = 0 
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append((img, path))

            # Generate a unique filename to avoid conflicts
            filename = f"{count}_{os.path.basename(path)}"
            destination = os.path.join(target_path, filename)
            shutil.copy2(path, destination)

            count += 1
        except Exception as e:
            print(f"Failed to load or copy image {path}: {e}")

    print(f"Loaded and copied {len(images)} images to: {target_path}")

class LeNet(nn.Module):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.05, num_classes=2):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(in_features=2304, out_features=256), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(256, 84), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(84, 2)
        )
        
    def layer_summary(self, input_shape):
        X = torch.randn(input_shape)
        for i, layer in enumerate(self.net):
            try:
                X = layer(X)
            except Exception as e:
                print(f"Error at layer {i}: {layer}")
                print(f"Input shape: {X.shape}")
                print(f"Error message: {e}")
               
    def forward(self, x):
        return self.net(x)



def binary_classification(): 
    print("hi in binary classification")
    transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = CustomImageDataset(root_dir="/Users/aaryaamoharir/Desktop/Summer 2025 /Research /minorityML/combined/", transform=transform)
    
    #test out a smaller dataset size for now 
    ten_percent_size = int(0.1 * len(full_dataset))
    _, reduced_dataset = random_split(full_dataset, [len(full_dataset) - ten_percent_size, ten_percent_size])
    
    #do a 80/20 split for right now 

    train_size = int(0.8 * len(reduced_dataset))
    test_size = len(reduced_dataset) - train_size
    train_dataset, test_dataset = random_split(reduced_dataset, [train_size, test_size])

    #train_size = int(0.8 * len(full_dataset))
    #test_size = len(full_dataset) - train_size
    #train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    #only show the images in the train dataset
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    model = LeNet(num_classes=2)
    model.layer_summary((1, 3, 28, 28))

    #attempting to train model without d2l 
    loss_fn = torch.nn.CrossEntropyLoss()
    #lower learning rate 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    print("about to train epoch")
    #training loop 
    for epoch in range(20):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in loop:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")
    return model,test_loader

def eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")







if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #store it all in one single jpg 
    #convert_to_one_directory("/Users/aaryaamoharir/Desktop/Summer 2025 /Research /minorityML/data/")
    model,test_loader = binary_classification()
    eval(model, test_loader)
