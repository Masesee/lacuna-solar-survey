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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_counter_model(args):
    """Train a model that counts solar panels and boilers"""
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
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader)  # Fix: use train_dataloader
        for batch in progress_bar:
            images = batch["image"].to(device)
            targets = batch["counts"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            panel_pred, boiler_pred = model(images)
            
            # Compute loss
            panel_loss = criterion(panel_pred.squeeze(), targets[:, 0])
            boiler_loss = criterion(boiler_pred.squeeze(), targets[:, 1])
            loss = panel_loss + boiler_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} [Train Loss: {loss.item():.4f}]")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        panel_mae = 0.0
        boiler_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):  # Fix: use val_dataloader
                images = batch["image"].to(device)
                targets = batch["counts"].to(device)
                
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
        
        val_loss /= len(val_loader)
        panel_mae /= len(val_loader)
        boiler_mae /= len(val_loader)
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
    train_loader, val_loader = get_dataloaders(
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
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader)
        for batch in progress_bar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} [Train Loss: {loss.item():.4f}]")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
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
    parser = argparse.ArgumentParser(description="Train solar panel detection models")
    parser.add_argument("--model_type", type=str, default="counter", 
                        choices=["counter", "segmentation"],
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
    
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the selected model
    if args.model_type == "counter":
        train_counter_model(args)
    elif args.model_type == "segmentation":
        train_segmentation_model(args)
    else:
        print(f"Unknown model type: {args.model_type}")
