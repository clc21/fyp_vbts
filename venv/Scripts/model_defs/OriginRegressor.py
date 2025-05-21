import torch.nn as nn
from torchvision.models import resnet18

class OriginRegressor(nn.Module):
    def __init__(self):
        super(OriginRegressor, self).__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, 2)  # output (x, y)
        self.model = base

    def forward(self, x):
        return self.model(x)
