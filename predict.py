import os
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def predict_counter(args):
    """Generate predictions using the counter model"""
    model = SolarPanelCounter().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.model_path}")

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

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch["image"].to(device)
            ids = batch["id"]

            try:
                panel_pred, boiler_pred = model(images)
            except Exception as e:
                print(f"Error during model inference for batch with IDs {ids}: {e}")
                continue

            panel_pred = torch.clamp(torch.round(panel_pred.squeeze()), min=0)
            boiler_pred = torch.clamp(torch.round(boiler_pred.squeeze()), min=0)

            for i in range(len(ids)):
                predictions.append({
                    'id': ids[i],
                    'pan_nbr': int(panel_pred[i].item()),
                    'boil_nbr': int(boiler_pred[i].item())
                })

    # Save predictions to CSV
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
        print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.csv')}")

        # Optional: Visualize distribution of predictions
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

    if args.mode == 'counter':
        predict_counter(args)
    elif args.mode == 'segmentation':
        predict_segmentation(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
