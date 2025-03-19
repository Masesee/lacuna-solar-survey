# Lacuna Solar Survey project

Dataset Link: [Dataset](https://www.kaggle.com/datasets/kimp1995/lacuna-solar-survey-challenge/data)

- train.py 
  1. ``python train.py --model_type counter --train_csv path/to/train.csv --img_dir path/to/images --output_dir ./output``
  2. ``python train.py --model_type segmentation --train_csv path/to/train.csv --img_dir path/to/images --output_dir ./output``

- evalaute.py
  - ``python evalaute.py --model_path ./output/best_counter_model.pth --train_csv ./data/train.csv --val_csv ./data/val.csv --img_dir ./images --batch_size 16 --img_size 512``