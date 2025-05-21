import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ShapeClassifier, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
