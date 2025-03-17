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
  try:
    return np.array(ast.literal_eval(polygon_string), np.int32)
  except (SyntaxError, ValueError):
    print(f"Failed to parse polygon: {polygon_str}")
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

  
