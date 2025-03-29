import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_loss_fn(self):
        """Return the loss function for this model"""
        raise NotImplementedError

    def get_optimizer(self, lr):
        """Return the optimizer for this model"""
        raise NotImplementedError

    def predict(self, x):
        """Return predictions in the required format"""
        raise NotImplementedError

    def get_metrics(self):
        """Return a dict of metrics to track during training"""
        raise NotImplementedError
