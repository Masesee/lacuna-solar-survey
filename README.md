# Lacuna Solar Survey project

Dataset Link: [Dataset](https://www.kaggle.com/datasets/kimp1995/lacuna-solar-survey-challenge/data)

- train.py 
  1. ``python train.py --model_type counter --train_csv ./data/train.csv --img_dir ./data/images --output_dir ./output --batch_size 16 --epochs 30 --learning_rate 1e-4 --img_size 512 --num_classes 2``
  2. ``python train.py --model_type segmentation --train_csv ./data/train.csv --img_dir ./data/images --output_dir ./output --batch_size 16 --epochs 30 --learning_rate 1e-4 --img_size 512 --num_classes 2``

- train_optimised.py (similar usage to train.py just edit name)

- evaluate.py
  - ``python evaluate.py --model_path ./output/best_counter_model.pth --train_csv ./data/train.csv --val_csv ./data/val.csv --img_dir ./images --batch_size 16 --img_size 512``

- predict.py
  - ``python predict.py --model_path ./output/best_counter_model.pth --test_csv ./data/test.csv --img_dir ./images --output_dir ./predictions --batch_size 16 --img_size 512``
