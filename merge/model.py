import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# from tianchi_densenet201_master import model_densenet201
# from tianchi_densenet169_master import model_densenet169
# from tianchi_inceptionv4_master import model_inceptionv4

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
#         self.model1 = model_densenet201.net()
#         self.model2 = model_densenet169.densenet169()
#         self.model2 = model_inceptionv4.v4()
#         #最后的连接层
        self.fc = nn.Linear(36, 12)
    def forward(self, x):
#         x1 = self.model1(x)
#         x2 = self.model1(x)
#         x3 = self.model1(x)
#         end = torch.cat((x1, x2, x3), 1)
        
        x = self.fc(x)
        return x
