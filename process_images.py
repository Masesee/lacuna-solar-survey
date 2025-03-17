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
def process_data(csv_path, output_dir, display=False):
  """Preprocess a dataset by loading images, parsing annotations, drawing them, and saving the results"""
  df = pd.read_csv(csv_path)

  for _, row in df.iterrows():
    image_name = row['ID'] + '.jpg'
    img_path = os.path.join(IMAGE_DIR, image_name)

    polygon_coords = parse_polygon(row['polygon'])
    if polygon_coords is None:
      continue

    annotations = (polygon_coords, row['pan_nbr'], row['boil_nbr'], row['placement'])

    image = load_image(img_path)
    if img_path is None:
      continue

    annotated_image = draw_annotations(image, annotations)

    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if display:
      plt.figure(figsize=(10,10))
      plt.imshow(annotated_image)
      plt.axis('off')
      plt.show()

if __name__ == "__main__":
  # Process training dataset
  print('Processing training dataset...')
  process_dataset(TRAIN_CSV_PATH, OUTPUT_TRAIN_DIR, display=False)

  # Process test dataset
  print("Processing test dataset...")
  process_dataset(TEST_CSV_PATH, OUTPUT_TEST_DIR, display=False)
