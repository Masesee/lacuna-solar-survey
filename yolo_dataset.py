import os
import shutil
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

def convert_to_yolo_format(polygon_str, img_width, img_height):
    """Convert polygon coordinates to YOLO format (x_center, y_center, width, height)"""
    # Parse polygon string to coordinates
    coords = np.array([float(x) for x in polygon_str.strip('[]').split(',')]).reshape(-1, 2)
    polygon = Polygon(coords)
    
    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds
    
    # Convert to YOLO format (normalized coordinates)
    x_center = ((minx + maxx) / 2) / img_width
    y_center = ((miny + maxy) / 2) / img_height
    width = (maxx - minx) / img_width
    height = (maxy - miny) / img_height
    
    return x_center, y_center, width, height

def prepare_yolo_dataset(csv_path, img_dir, output_dir, split_ratio=0.2):
    """
    Prepare dataset in YOLO format
    
    Args:
        csv_path: Path to CSV file with annotations
        img_dir: Directory containing images
        output_dir: Directory to save YOLO format dataset
        split_ratio: Train/val split ratio
    """
    # Create directory structure
    yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in yolo_dirs:
        os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
    
    # Read and process data
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    val_size = int(total_samples * split_ratio)
    
    # Create data.yaml file
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: 2
names: ['panel', 'boiler']
    """
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # Process each image and its annotations
    for idx, row in df.iterrows():
        # Determine if sample goes to train or val set
        is_val = idx < val_size
        subset = 'val' if is_val else 'train'
        
        # Copy image
        img_name = f"{row['ID']}.jpg"
        src_img_path = os.path.join(img_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'images', subset, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Create label file
        label_path = os.path.join(output_dir, 'labels', subset, f"{row['ID']}.txt")
        
        # Get image dimensions
        img = cv2.imread(src_img_path)
        img_height, img_width = img.shape[:2]
        
        # Process annotations
        try:
            if pd.notna(row['polygon']):
                # Convert polygon to YOLO format
                x_center, y_center, width, height = convert_to_yolo_format(
                    row['polygon'], img_width, img_height)
                
                # Write label file (0 for panel)
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
                
                # If there's a boiler, add it as class 1
                if row['boil_nbr'] > 0:
                    # Assuming similar dimensions for boiler
                    f.write(f"1 {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            print(f"Error processing {row['ID']}: {e}")
            continue
    
    print(f"Dataset prepared in YOLO format at {output_dir}")
    return os.path.join(output_dir, 'data.yaml')
