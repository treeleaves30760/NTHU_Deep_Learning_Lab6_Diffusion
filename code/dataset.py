import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ICLEVRDataset(Dataset):
    def __init__(self, json_path, image_dir=None, transform=None, is_train=True):
        """
        Args:
            json_path (str): Path to the json file with annotations
            image_dir (str, optional): Path to the image directory
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Whether this is training set or testing set
        """
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load object dictionary
        with open('objects.json', 'r') as f:
            self.object_dict = json.load(f)

        # Load data json
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle data structure - ensure it's a list of conditions
        if isinstance(data, dict):
            # If data is a dictionary, store it as a list of tuples (filename, conditions)
            self.data = [(img_filename, conditions)
                         for img_filename, conditions in data.items()]
            self.data.sort()  # Sort by filename
        else:
            # If data is already a list, use it directly
            self.data = [(f"idx_{i}", conditions)
                         for i, conditions in enumerate(data)]

        self.num_classes = len(self.object_dict)
        self.is_train = is_train
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = torch.zeros(self.num_classes)
        img_filename, conditions = self.data[idx]

        # Convert conditions to one-hot encoding
        for condition in conditions:
            labels[self.object_dict[condition]] = 1.0

        # For training data, load the corresponding image
        if self.is_train and self.image_dir:
            try:
                img_path = os.path.join(self.image_dir, img_filename)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                return image, labels
            except Exception as e:
                print(f"Error loading image {img_filename}: {e}")
                # Return a blank image in case of error
                dummy_image = torch.zeros(3, 64, 64)
                return dummy_image, labels

        # If not training or no image_dir, return a dummy image for consistent output shape
        dummy_image = torch.zeros(3, 64, 64)
        return dummy_image, labels


def get_dataloader(json_path, image_dir=None, batch_size=32, is_train=True, shuffle=True):
    """
    Create data loaders for training or testing
    """
    dataset = ICLEVRDataset(json_path, image_dir, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=4)
    return dataloader
