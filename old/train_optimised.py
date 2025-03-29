import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import local modules
from model_architectures import SolarPanelCounter, SolarSegmentationModel, get_faster_rcnn_model
from dataset import get_dataloaders, get_test_dataloader

# Set CUDA optimization flags
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Enhanced GPU check and setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

# Add memory management
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_model(model_type, **kwargs):
    """Factory function to create models"""
    if model_type == "counter":
        return SolarPanelCounter(**kwargs)
    elif model_type == "segmentation":
        return SolarSegmentationModel(**kwargs)
    elif model_type == "yolo":
        return YOLOSolarDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(args):
    """Generic training function for any model type"""
    clear_gpu_memory()
    
    if args.model_type == "yolo":
        # Prepare YOLO dataset
        print("Preparing YOLO dataset...")
        yolo_data_yaml = prepare_yolo_dataset(
            args.train_csv,
            args.img_dir,
            os.path.join(args.output_dir, 'yolo_dataset'),
            split_ratio=args.val_split
        )
        args.data_yaml = yolo_data_yaml
    
    # Initialize model
    model = get_model(args.model_type, **vars(args))
    model = model.to(device)
    
    # Get model-specific components
    criterion = model.get_loss_fn()
    optimizer = model.get_optimizer(args.learning_rate)
    metrics = model.get_metrics()
    
    if args.model_type == "yolo":
        # Use YOLO's native training method
        model.train_model(args.data_yaml)
    else:
        # Use custom training loop
        if args.model_type == "counter":
            train_counter_model(args)
        elif args.model_type == "segmentation":
            train_segmentation_model(args)

def train_counter_model(args):
    """Train a model that counts solar panels and boilers"""
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # Split the data into training and validation sets
    df = pd.read_csv(args.train_csv)

    # Check if validation csv is provided
    if args.val_csv and os.path.exists(args.val_csv):
        train_df = df
        val_df = pd.read_csv(args.val_csv)
    else:
        print("Splitting the training data into training and validation sets...")
        train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)

        # Save the splits
        train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
        val_df.to_csv(os.path.join(args.output_dir, "val_split.csv"), index=False)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Create dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        os.path.join(args.output_dir, "train_split.csv"),
        os.path.join(args.output_dir, "val_split.csv"),
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="counter",
        img_size=args.img_size
        )
    
    # Initialize the model
    print("Initializing the model...")
    model = SolarPanelCounter(num_classes=args.num_classes).to(device)  # Pass num_classes here

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()  # Fix typo: should be nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    try:
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_dataloader)  # Fix: use train_dataloader
            for batch in progress_bar:
                images = batch["image"].to(device, non_blocking=True)  # Add non_blocking=True
                targets = batch["counts"].to(device, non_blocking=True)
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    # Forward pass
                    optimizer.zero_grad(set_to_none=True)  # More efficient than False
                    panel_pred, boiler_pred = model(images)
                    
                    # Compute loss
                    panel_loss = criterion(panel_pred.squeeze(), targets[:, 0])
                    boiler_loss = criterion(boiler_pred.squeeze(), targets[:, 1])
                    loss = panel_loss + boiler_loss
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} [Train Loss: {loss.item():.4f}]")
            
            train_loss /= len(train_dataloader)  # Fix: use train_dataloader
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            panel_mae = 0.0
            boiler_mae = 0.0
            
            with torch.no_grad(), torch.cuda.amp.autocast():  # Add autocast here too
                for batch in tqdm(val_dataloader, desc="Validating"):  # Fix: use val_dataloader
                    images = batch["image"].to(device, non_blocking=True)
                    targets = batch["counts"].to(device, non_blocking=True)
                    
                    # Forward pass
                    panel_pred, boiler_pred = model(images)
                    
                    # Compute loss
                    panel_loss = criterion(panel_pred.squeeze(), targets[:, 0])
                    boiler_loss = criterion(boiler_pred.squeeze(), targets[:, 1])
                    loss = panel_loss + boiler_loss
                    
                    # Compute MAE
                    panel_mae += torch.abs(panel_pred.squeeze() - targets[:, 0]).mean().item()
                    boiler_mae += torch.abs(boiler_pred.squeeze() - targets[:, 1]).mean().item()
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)  # Fix: use val_dataloader
            panel_mae /= len(val_dataloader)  # Fix: use val_dataloader
            boiler_mae /= len(val_dataloader)  # Fix: use val_dataloader
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Panel MAE: {panel_mae:.4f}, Boiler MAE: {boiler_mae:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(args.output_dir, 'best_counter_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise e
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, os.path.join(args.output_dir, 'final_counter_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    
    print("Training completed!")

def train_segmentation_model(args):
    """Train a segmentation model for solar panel regions"""
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    print("Loading data...")
    # Split data into train and validation
    df = pd.read_csv(args.train_csv)
    
    # Check if validation csv exists, else do a random split
    if args.val_csv and os.path.exists(args.val_csv):
        train_df = df
        val_df = pd.read_csv(args.val_csv)
    else:
        print("Splitting data into train and validation sets...")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save the splits
        train_df.to_csv(os.path.join(args.output_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(args.output_dir, 'val_split.csv'), index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Create dataloaders
    train_dataloader, val_dataloader = get_dataloaders(  # Fix: rename variables
        os.path.join(args.output_dir, 'train_split.csv'), 
        os.path.join(args.output_dir, 'val_split.csv'),
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="segmentation",
        img_size=args.img_size
    )
    
    # Initialize model
    print("Initializing model...")
    model = SolarSegmentationModel(n_classes=1).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    try:
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_dataloader)  # Fix: use train_dataloader
            for batch in progress_bar:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    # Forward pass
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    
                    # Compute loss
                    loss = criterion(outputs, masks)
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} [Train Loss: {loss.item():.4f}]")
            
            train_loss /= len(train_dataloader)  # Fix: use train_dataloader
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):  # Fix: use val_dataloader
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Compute loss
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)  # Fix: use val_dataloader
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(args.output_dir, 'best_segmentation_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise e
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, os.path.join(args.output_dir, 'final_segmentation_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    
    print("Training completed!")

if __name__ == "__main__":
    # Add GPU-specific arguments
    parser = argparse.ArgumentParser(description="Train solar panel detection models")
    parser.add_argument("--model_type", type=str, default="counter", 
                        choices=["counter", "segmentation", "yolo"],
                        help="Model type to train (counter or segmentation)")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="",
                        help="Path to validation CSV file (optional)")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./model_output",
                        help="Directory to save model and results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size for resizing")
    parser.add_argument("--num_classes", type=int, default=2,  # Add num_classes argument
                        help="Number of output classes for the model")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proportion of the dataset to use for validation (if val_csv is not provided)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--pin_memory", type=bool, default=True,
                        help="Pin memory for faster GPU transfer")
    parser.add_argument("--data_yaml", type=str, default="",
                        help="Path to YOLO data.yaml file (required for YOLO training)")
    parser.add_argument("--model_size", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size (nano to extra large)")
    
    args = parser.parse_args()
    
    # Verify CUDA is available and working
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
        except RuntimeError as e:
            print(f"CUDA initialization failed: {e}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the selected model
    train_model(args)
