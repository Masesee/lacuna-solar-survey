import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
from model_architectures import SolarPanelCounter, SolarSegmentationModel
from dataset import get_test_dataloader

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

def clear_gpu_memory():
    """Clear GPU memory and reset stats"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def predict_counter(args):
    """Generate predictions using the counter model"""
    clear_gpu_memory()
    
    try:
        # Initialize model with proper num_classes
        model = SolarPanelCounter(num_classes=2).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_loader = get_test_dataloader(
        args.test_csv,
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="counter",
        img_size=args.img_size
    )

    # Check if the dataset is empty
    if len(test_loader.dataset) == 0:
        print(f"Test dataset is empty or improperly formatted. Please check the test CSV at {args.test_csv} and image directory at {args.img_dir}.")
        return

    predictions = []
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                images = batch["image"].to(device, non_blocking=True)
                ids = batch["id"]

                # Forward pass
                panel_pred, boiler_pred = model(images)
                
                # Ensure synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                panel_pred = torch.clamp(torch.round(panel_pred.squeeze()), min=0)
                boiler_pred = torch.clamp(torch.round(boiler_pred.squeeze()), min=0)

                for i in range(len(ids)):
                    predictions.append({
                        'id': ids[i],
                        'pan_nbr': int(panel_pred[i].item()),
                        'boil_nbr': int(boiler_pred[i].item())
                    })

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory")
            clear_gpu_memory()
        raise e

    # Save predictions to CSV
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.csv')}")

        # Visualize distribution of predictions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(pred_df['pan_nbr'], bins=20)
        plt.xlabel('Panel Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Panel Counts')

        plt.subplot(1, 2, 2)
        plt.hist(pred_df['boil_nbr'], bins=20)
        plt.xlabel('Boiler Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Boiler Counts')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'prediction_distribution.png'))
        print(f"Prediction distribution plot saved to {os.path.join(args.output_dir, 'prediction_distribution.png')}")
    else:
        print("No predictions were generated. Please check the model and test data.")

def predict_segmentation(args):
    """Generate predictions using the segmentation model"""
    clear_gpu_memory()
    
    try:
        model = SolarSegmentationModel(n_classes=1).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_loader = get_test_dataloader(
        args.test_csv,
        args.img_dir,
        batch_size=args.batch_size,
        dataset_type="segmentation",
        img_size=args.img_size
    )

    os.makedirs(args.output_dir, exist_ok=True)
    predictions = []

    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                images = batch["image"].to(device, non_blocking=True)
                ids = batch["id"]

                # Forward pass
                masks = model(images)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Process predictions
                masks = torch.sigmoid(masks) > 0.5
                masks = masks.cpu().numpy()

                for i, mask in enumerate(masks):
                    mask_path = os.path.join(args.output_dir, f"{ids[i]}_mask.png")
                    cv2.imwrite(mask_path, (mask[0] * 255).astype(np.uint8))
                    predictions.append({
                        'id': ids[i],
                        'mask_path': mask_path
                    })

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory")
            clear_gpu_memory()
        raise e

    # Save predictions metadata
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(args.output_dir, 'segmentation_predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(args.output_dir, 'segmentation_predictions.csv')}")
    else:
        print("No predictions were generated. Please check the model and test data.")

def main():
    parser = argparse.ArgumentParser(description="Solar Panel Detection and Counting")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for model input')
    parser.add_argument('--mode', type=str, choices=['counter', 'segmentation'], required=True,
                        help='Mode: counter or segmentation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify CUDA is available and working
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
        except RuntimeError as e:
            print(f"CUDA initialization failed: {e}")
            sys.exit(1)

    if args.mode == 'counter':
        predict_counter(args)
    elif args.mode == 'segmentation':
        predict_segmentation(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
