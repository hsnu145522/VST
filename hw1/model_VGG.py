import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# You can add any other package, class and function if you need.

cfg = {
    'myVGG': [4, 'M', 8, 'M', 16, 'M', 32, "M", 32, "M"]
}

class ClassificationModel(nn.Module): # Don't modify the name of this class.
    def __init__(self, vgg_name="myVGG"):
        super(ClassificationModel, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        self.fc = nn.Sequential(
            nn.Linear(1568, 100),
            )
        self._initialize_weight()
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    model = ClassificationModel().to(torch.device(device))
    print(model)
