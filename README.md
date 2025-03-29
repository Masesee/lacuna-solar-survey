# Lacuna Solar Survey project
A Python-based tool for processing and preparing solar panel image datasets for YOLO object detection training.

## Project Overview

This project prepares a dataset of solar panel images and their polygon annotations for training YOLO object detection models. It converts polygon coordinates to YOLO format (normalized bounding box coordinates) and organizes the dataset into training and validation sets.

## Features

- Converts polygon annotations
- Automatically splits dataset into training and validation sets
- Handles various polygon coordinate formats
- Preserves aspect ratios and image dimensions
- Built-in validation and error handling

## Wiki
#### 1. annotation_utils.py
The code is part of a larger system for processing solar panel annotations, where polygons represent the boundaries of solar panels in images.(It's used within the dataset.py script)

#### 2. dataset.py
This code sets up a data preprocessing pipeline for a solar panel detection project. It's designed to verify data structure and prepare images with their annotations for model training.

**Purpose**
- Process train and test images with their annotations
- Draw annotations on images
- Save annotated images to output directories


## Dataset
Dataset Link: [Dataset](https://www.kaggle.com/datasets/kimp1995/lacuna-solar-survey-challenge/data)

## Example Usage
- train.py 
  1. ``python train.py --model_type counter --train_csv ./data/train.csv --img_dir ./data/images --output_dir ./output --batch_size 16 --epochs 30 --learning_rate 1e-4 --img_size 512 --num_classes 2``

  2. ``python train.py --model_type segmentation --train_csv ./data/train.csv --img_dir ./data/images --output_dir ./output --batch_size 16 --epochs 30 --learning_rate 1e-4 --img_size 512 --num_classes 2``

- train_optimised.py
  - ``python train_optimised.py --model_type counter --train_csv ./data/train.csv --img_dir ./data/images --output_dir ./output --batch_size 16 --epochs 30 --learning_rate 1e-4 --img_size 512 --num_classes 2 --val_split 0.2 --seed 42 --num_workers 4 --pin_memory True``

- evaluate.py
  - ``python evaluate.py --model_path ./output/best_counter_model.pth --train_csv ./data/train.csv --val_csv ./data/val.csv --img_dir ./images --batch_size 16 --img_size 512``

- predict.py
  - ``python predict.py --model_path ./output/best_segmentation_model.pth --test_csv ./data/test.csv --img_dir ./data/images --output_dir ./predictions --batch_size 16 --img_size 512 --mode segmentation``

- predict_optimised.py
  - ``python predict_optimised.py --model_path ./output/best_counter_model.pth --test_csv ./data/test.csv --img_dir ./data/images --output_dir ./predictions --batch_size 16 --img_size 512 --mode counter --submission_name counter_submission.csv``


## under review (*****)
- Train counter model
python train_optimised.py --model_type counter --train_csv train.csv

- Train YOLO model
python train_optimised.py --model_type yolo --data_yaml data.yaml --model_size s