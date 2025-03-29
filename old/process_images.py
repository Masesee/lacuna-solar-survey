import os
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from annotations_utils import load_image, parse_polygon, draw_annotations

# PATHS
BASE_PATH = '/kaggle/input/lacuna-solar-survey-challenge'
IMAGE_DIR = os.path.join(BASE_PATH, "images")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "Train.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "Test.csv")

# Output directories (must be in /kaggle/working/)
OUTPUT_TRAIN_DIR = "/kaggle/working/annotated_train"
OUTPUT_TEST_DIR = "/kaggle/working/annotated_test"

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

# Preprocess dataset using csv
def process_data(csv_path, output_dir, is_test=False, display=False):
    """Preprocess a dataset by loading images, parsing annotations, drawing them, and saving the results"""
    df = pd.read_csv(csv_path)
    
    # Debug: Print columns and first few rows
    print(f"Columns in CSV: {df.columns.tolist()}")
    print(f"First few rows: {df.head(1)}")
    
    # Check which columns are available
    has_polygon = 'polygon' in df.columns
    
    if has_polygon:
        # Debug: Print sample polygon values
        print("Sample polygon values:")
        for i, poly in enumerate(df['polygon'].head(3)):
            print(f"  {i}: {poly} (type: {type(poly)})")
    
    # Track progress
    total = len(df)
    success = 0
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            image_name = row['ID'] + '.jpg'
            img_path = os.path.join(IMAGE_DIR, image_name)
            
            # Load the image first
            image = load_image(img_path)
            if image is None:
                errors += 1
                continue
            
            # For test data, we may not have polygon annotations
            if is_test and not has_polygon:
                # Just save the original image
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                success += 1
                continue
            
            # Process annotations if available
            polygon_coords = parse_polygon(row['polygon']) if has_polygon else None
            
            # For training data, skip images without valid polygons
            if not is_test and polygon_coords is None:
                errors += 1
                continue
                
            # Get other annotation fields if available
            pan_nbr = row.get('pan_nbr', 0)
            boil_nbr = row.get('boil_nbr', 0)
            placement = row.get('placement', 'unknown')
            
            # Draw annotations if we have them
            if polygon_coords is not None:
                annotations = (polygon_coords, pan_nbr, boil_nbr, placement)
                annotated_image = draw_annotations(image, annotations)
            else:
                annotated_image = image
            
            if annotated_image is None:
                errors += 1
                continue
                
            # Save the annotated image
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            # Display if requested
            if display:
                plt.figure(figsize=(10,10))
                plt.imshow(annotated_image)
                plt.axis('off')
                plt.title(f"ID: {row['ID']}")
                plt.show()
                
            success += 1
            
            # Print progress periodically
            if (idx + 1) % 100 == 0 or idx == total - 1:
                print(f"Processed {idx + 1}/{total} images. Success: {success}, Errors: {errors}")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            errors += 1
    
    print(f"Processing complete. Total: {total}, Success: {success}, Errors: {errors}")

if __name__ == "__main__":
    # Process training dataset
    print('Processing training dataset...')
    process_data(TRAIN_CSV_PATH, OUTPUT_TRAIN_DIR, is_test=False, display=False)
    
    # Process test dataset
    print("Processing test dataset...")
    process_data(TEST_CSV_PATH, OUTPUT_TEST_DIR, is_test=True, display=False)
