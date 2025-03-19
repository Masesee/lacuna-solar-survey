import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import local modules
from model_architectures import SolarPanelCounter, SolarSegmentationModel
from dataset import get_dataloaders 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate_counter_model(args):
    """Evaluate the counter model's performance"""
    # Load the model
    model = SolarPanelCounter().to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    
    # Load validation data
    _, val_loader = get_dataloaders(
        args.train_csv, 
        args.val_csv,
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="counter",
        img_size=args.img_size
    )
    
    # Evaluation metrics
    panel_preds = []
    panel_targets = []
    boiler_preds = []
    boiler_targets = []
    
    # Evaluate model
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            targets = batch["counts"].to(device)
            
            # Forward pass
            panel_pred, boiler_pred = model(images)
            
            # Round predictions to nearest integer
            panel_pred = torch.round(panel_pred.squeeze())
            boiler_pred = torch.round(boiler_pred.squeeze())
            
            # Collect predictions and targets
            panel_preds.extend(panel_pred.cpu().numpy())
            panel_targets.extend(targets[:, 0].cpu().numpy())
            boiler_preds.extend(boiler_pred.cpu().numpy())
            boiler_targets.extend(targets[:, 1].cpu().numpy())
    
    # Calculate metrics
    panel_mae = mean_absolute_error(panel_targets, panel_preds)
    panel_rmse = np.sqrt(mean_squared_error(panel_targets, panel_preds))
    panel_r2 = r2_score(panel_targets, panel_preds)
    
    boiler_mae = mean_absolute_error(boiler_targets, boiler_preds)
    boiler_rmse = np.sqrt(mean_squared_error(boiler_targets, boiler_preds))
    boiler_r2 = r2_score(boiler_targets, boiler_preds)
    
    print("Solar Panel Detection Results:")
    print(f"Panel MAE: {panel_mae:.4f}")
    print(f"Panel RMSE: {panel_rmse:.4f}")
    print(f"Panel R²: {panel_r2:.4f}")
    print("\nSolar Boiler Detection Results:")
    print(f"Boiler MAE: {boiler_mae:.4f}")
    print(f"Boiler RMSE: {boiler_rmse:.4f}")
    print(f"Boiler R²: {boiler_r2:.4f}")
    
    # Plot predictions vs targets
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(panel_targets, panel_preds, alpha=0.5)
    plt.plot([0, max(panel_targets)], [0, max(panel_targets)], 'r--')
    plt.xlabel('Actual Panel Count')
    plt.ylabel('Predicted Panel Count')
    plt.title('Solar Panel Count Predictions')
    
    plt.subplot(1, 2, 2)
    plt.scatter(boiler_targets, boiler_preds, alpha=0.5)
    plt.plot([0, max(boiler_targets)], [0, max(boiler_targets)], 'r--')
    plt.xlabel('Actual Boiler Count')
    plt.ylabel('Predicted Boiler Count')
    plt.title('Solar Boiler Count Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'prediction_plots.png'))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'id': np.arange(len(panel_targets)),
        'actual_panel': panel_targets,
        'predicted_panel': panel_preds,
        'actual_boiler': boiler_targets,
        'predicted_boiler': boiler_preds
    })
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    # Save metrics to a text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Solar Panel Detection Results:\n")
        f.write(f"Panel MAE: {panel_mae:.4f}\n")
        f.write(f"Panel RMSE: {panel_rmse:.4f}\n")
        f.write(f"Panel R²: {panel_r2:.4f}\n\n")
        f.write("Solar Boiler Detection Results:\n")
        f.write(f"Boiler MAE: {boiler_mae:.4f}\n")
        f.write(f"Boiler RMSE: {boiler_rmse:.4f}\n")
        f.write(f"Boiler R²: {boiler_r2:.4f}\n")

def evaluate_segmentation_model(args):
    """Evaluate the segmentation model's performance"""
    # Load the model
    model = SolarSegmentationModel(n_classes=1).to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    
    # Load validation data
    _, val_loader = get_dataloaders(
        args.train_csv, 
        args.val_csv,
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="segmentation",
        img_size=args.img_size
    )
    
    # Evaluation metrics
    dice_scores = []
    iou_scores = []
    
    # Evaluate model
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Convert to binary masks
            preds = (probs > 0.5).float()
            
            # Calculate metrics
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # Calculate Dice coefficient
                intersection = np.sum(pred * mask)
                dice = (2. * intersection) / (np.sum(pred) + np.sum(mask) + 1e-6)
                dice_scores.append(dice)
                
                # Calculate IoU
                union = np.sum(pred) + np.sum(mask) - intersection
                iou = intersection / (union + 1e-6)
                iou_scores.append(iou)
    
    # Calculate average metrics
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    print("Segmentation Results:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    
    # Save metrics to a text file
    with open(os.path.join(args.output_dir, 'segmentation_metrics.txt'), 'w') as f:
        f.write("Segmentation Results:\n")
        f.write(f"Average Dice Score: {avg_dice:.4f}\n")
        f.write(f"Average IoU Score: {avg_iou:.4f}\n")
    
    # Visualize some predictions
    num_samples = min(5, len(val_loader.dataset))
    _, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))  # Removed unused variable 'fig'
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = val_loader.dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].cpu().numpy()[0]
            
            # Get prediction
            output = model(image)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred = (prob > 0.5).astype(np.float32)
            
            # Display original image
            img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_np = img_np.astype(np.uint8)
            
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # Display ground truth mask
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Display predicted mask
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'segmentation_examples.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate solar panel detection models")
    parser.add_argument("--model_type", type=str, default="counter", 
                        choices=["counter", "segmentation"],
                        help="Model type to evaluate (counter or segmentation)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV file")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size for resizing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate the selected model
    if args.model_type == "counter":
        evaluate_counter_model(args)
    elif args.model_type == "segmentation":
        evaluate_segmentation_model(args)
    else:
        print(f"Unknown model type: {args.model_type}")