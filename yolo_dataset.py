import os
import shutil
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from PIL import Image

def convert_to_yolo_format(polygon_str, img_width, img_height):
    """Convert polygon coordinates to YOLO format (x_center, y_center, width, height)"""
    coords = np.array([float(x) for x in polygon_str.strip('[]').split(',')]).reshape(-1, 2)
    polygon = Polygon(coords)
    minx, miny, maxx, maxy = polygon.bounds
    x_center = ((minx + maxx) / 2) / img_width
    y_center = ((miny + maxy) / 2) / img_height
    width = (maxx - minx) / img_width
    height = (maxy - miny) / img_height
    return x_center, y_center, width, height

def prepare_yolo_dataset(csv_path, img_dir, output_dir, split_ratio=0.2):
    """Prepare dataset in YOLO format"""
    # Create directory structure
    yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in yolo_dirs:
        os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
    
    # Read and process data
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    val_size = int(total_samples * split_ratio)
    print(f"Total samples: {total_samples}, Validation size: {val_size}")

    # Split dataset
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    processed_count = 0
    for split, split_df in zip(['train', 'val'], [train_df, val_df]):
        for idx, row in split_df.iterrows():
            img_id = row['ID']
            polygon_str = row['polygon']
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            try:
                # Get image dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Copy image to YOLO directory
                shutil.copy(img_path, os.path.join(output_dir, f'images/{split}', f"{img_id}.jpg"))
                
                # Convert polygon to YOLO format
                x_center, y_center, width, height = convert_to_yolo_format(polygon_str, img_width, img_height)
                label_path = os.path.join(output_dir, f'labels/{split}', f"{img_id}.txt")
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images")
            except Exception as e:
                print(f"Error processing {img_id}: {str(e)}")

    print(f"Successfully processed {processed_count} images")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with annotations')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save YOLO dataset')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Train/val split ratio')
    args = parser.parse_args()

    prepare_yolo_dataset(args.csv_path, args.img_dir, args.output_dir, args.split_ratio)