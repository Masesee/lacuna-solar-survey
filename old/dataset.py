import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from shapely.geometry import Polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

class SolarPanelDataset(Dataset):
    """Dataset class for solar panel counting and placement detection"""
    def __init__(self, csv_path, img_dir, transform=None, test_mode=False):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transforms to be applied on a sample.
            test_mode (bool): If True, doesn't expect annotations columns.
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.test_mode = test_mode

        # Print dataset info
        print(f"Dataset created with {len(self.data)} images")
        if not test_mode and 'pan_nbr' in self.data.columns and 'boil_nbr' in self.data.columns:
            print(f"Distribution of panel counts: {self.data['pan_nbr'].value_counts().sort_index()}")
            print(f"Distribution of boiler counts: {self.data['boil_nbr'].value_counts().sort_index()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            if self.test_mode:
                return {'image': torch.zeros(3, 512, 512), 'id': img_id}
            else:
                return {'image': torch.zeros(3, 512, 512), 
                        'counts': torch.tensor([0, 0], dtype=torch.float32),
                        'id': img_id}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For test mode, return only the image and ID
        if self.test_mode:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            return {'image': image, 'id': img_id}
        
        # For training mode, get the targets
        panel_count = self.data.iloc[idx]['pan_nbr']
        boiler_count = self.data.iloc[idx]['boil_nbr']

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        counts = torch.tensor([panel_count, boiler_count], dtype=torch.float32)

        return {'image': image, 
                'counts': counts, 
                'id': img_id}

def get_transforms(mode='train', img_size=512):
    """Get transforms for training and validation/testing"""
    if (mode == 'train'):
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_dataloaders(train_csv, val_csv, img_dir, batch_size=8, dataset_type="counter", img_size=512, seed=42):
    """
    Create train and validation dataloaders from separate CSV files.
    """
    set_seed(seed)

    if dataset_type == "counter":
        train_dataset = SolarPanelDataset(train_csv, img_dir, transform=get_transforms('train', img_size))
        val_dataset = SolarPanelDataset(val_csv, img_dir, transform=get_transforms('val', img_size))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Add this for better GPU utilization
        prefetch_factor=2  # Add this for better data loading
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Add this for better GPU utilization
        prefetch_factor=2  # Add this for better data loading
    )

    return train_loader, val_loader

def get_test_dataloader(test_csv, img_dir, batch_size=16, dataset_type="counter", img_size=512):
    """
    Create test dataloader
    """
    if dataset_type == "counter":
        test_dataset = SolarPanelDataset(
            test_csv, 
            img_dir,
            transform=get_transforms('test', img_size),
            test_mode=True
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader
