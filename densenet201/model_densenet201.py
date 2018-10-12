import torchvision.models as models
import torch
import torch.nn as nn
import math
def net(classes=12):
    model=models.densenet201(pretrained=True)
    model.classifier = nn.Linear(69120, classes)
    return model