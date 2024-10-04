import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MBConvBlock, self).__init__()
        self.expand = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.bn1 = nn.BatchNorm2d(in_channels * 3)
        self.depthwise = nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size, stride, padding=1, groups=in_channels * 3)
        self.bn2 = nn.BatchNorm2d(in_channels * 3)
        self.squeeze = nn.Conv2d(in_channels * 3, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.expand(x)))
        x = F.relu(self.bn2(self.depthwise(x)))
        x = self.bn3(self.squeeze(x))
        if x.size() == identity.size():
            x = x + identity
        return x

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=100):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.mb1 = MBConvBlock(4, 8, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d(56)
        self.mb2 = MBConvBlock(8, 16, 3, 1)
        self.pool2 = nn.AdaptiveAvgPool2d(28) 
        self.mb3 = MBConvBlock(16, 32, 3, 1)
        self.pool3 = nn.AdaptiveAvgPool2d(14) 
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.pool4 = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mb1(x)
        x = self.pool(x)
        x = self.mb2(x)
        x = self.pool2(x)
        x = self.mb3(x)
        x = self.pool3(x)
        x = self.conv2(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    model = ClassificationModel().to(torch.device(device))
    print(model)
