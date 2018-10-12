import torchvision.models as models
import torch
import torch.nn as nn
import math

def restnet152(classes=12):
    model=models.resnet152(pretrained=True)
    
    model.fc = nn.Linear(73728, classes)
    return model