import torchvision.models as models
import torch
import torch.nn as nn
import math
def densenet169(classes=12):
    model=models.densenet169(pretrained=True)
    model.classifier = nn.Linear(59904, classes)
    return model