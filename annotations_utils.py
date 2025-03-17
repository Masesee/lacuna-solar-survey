import cv2
import os
import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Function to load an image 
def load_image(img_path):
  """Loads an image from a given path and converts it from BGR to RGB."""
  if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    return None
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

# Convert polygon string to list to a proper list of tuples
def parse_polygon(polygon_string):
    """Parses a string representation of a polygon into a Numpy array"""
    if pd.isna(polygon_string) or not isinstance(polygon_string, str):
        print(f"Invalid polygon data: {polygon_string}")
        return None
        
    try:
        # Clean the string and ensure it's properly formatted
        clean_str = polygon_string.strip()
        if not (clean_str.startswith('[') and clean_str.endswith(']')):
            print(f"Malformed polygon string: {polygon_string}")
            return None
            
        parsed_data = ast.literal_eval(clean_str)
        
        # Verify it's a list of coordinate pairs
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in parsed_data):
            print(f"Polygon data is not a list of coordinate pairs: {parsed_data}")
            return None
            
        return np.array(parsed_data, np.int32)
        
    except (SyntaxError, ValueError) as e:
        print(f"Failed to parse polygon: {polygon_string}")
        print(f"Error: {e}")
        return None

# Draw polygons and labels
def draw_annotations(image, annotations):
  """Draw polygons and labels on an image"""
  if annotations is None or len(annotations) == 0:
    print("No annnotations for this image")
    return image

  polygon_coords, nbr_pan, nbr_boil, placement = annotations

  if len(polygon_coords) < 3:
    print("Fewer than 3 points...")
    return image

  cv2.polylines(image, [polygon_coords.reshape((-1, 1, 2))], isClosed = True, color=(255,0,0), thickness=2)

  poly = Polygon(polygon_coords)
  centroid_x, centroid_y = map(int, poly.centroid.coords[0])

  label = f"P: {nbr_pan}, B: {nbr_boil}, {placement}"
  cv2.putText(image, label, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

  return image

  
