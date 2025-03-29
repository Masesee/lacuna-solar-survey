import torch
import torch.nn as nn
from ultralytics import YOLO
from base_model import BaseModel

class YOLOSolarDetector(BaseModel):
    def __init__(self, pretrained=True, model_size='s'):
        super().__init__()
        # Initialize YOLO model
        if pretrained:
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            self.model = YOLO(f'yolov8{model_size}.yaml')
        
        # Configure model for solar panel detection
        self.model.model.head.nc = 2  # panels and boilers
        
    def forward(self, x):
        return self.model(x)
    
    def get_loss_fn(self):
        # YOLO handles loss internally
        return None
    
    def get_optimizer(self, lr):
        # YOLO handles optimization internally
        return None
    
    def predict(self, x):
        results = self.model(x)
        # Process YOLO results into required format
        predictions = []
        for result in results:
            boxes = result.boxes
            panel_count = len(boxes[boxes.cls == 0])
            boiler_count = len(boxes[boxes.cls == 1])
            predictions.append({
                'panel_count': panel_count,
                'boiler_count': boiler_count,
                'boxes': boxes.xyxy,
                'confidence': boxes.conf
            })
        return predictions
    
    def get_metrics(self):
        return {
            'map50': lambda x, y: self.model.metrics.map50,
            'map75': lambda x, y: self.model.metrics.map75
        }

    @staticmethod
    def load_from_checkpoint(path):
        model = YOLOSolarDetector(pretrained=False)
        model.model = YOLO(path)
        return model

    def train_model(self, data_yaml, epochs=100, batch_size=16, img_size=640):
        """
        Train YOLO model using its native training method
        
        Args:
            data_yaml: Path to data.yaml file for YOLO training
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
