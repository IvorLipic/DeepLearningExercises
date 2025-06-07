import torch
import torch.nn as nn

class CIFARModel(nn.Module):
    def __init__(self, in_channels=3, conv1_width=16, conv2_width=32,
                 fc1_width=256, fc2_width=128, class_count=10):
        super(CIFARModel, self).__init__()

        # Convolutional layers: conv(16,5) -> relu -> pool(3,2)
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Second convolutional block: conv(32,5) -> relu -> pool(3,2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Input: 32x32. After conv1 (with padding=2): 32x32
        # After pool1: (32-3)//2 + 1 = 15x15 
        # After conv2: 15x15
        # After pool2: (15-3)//2 + 1 = 7x7
        # flatten dimension = 32 channels * 7 * 7 = 1568
        self.flatten_dim = conv2_width * 7 * 7

        # Fully connected layers: fc(256) -> relu -> fc(128) -> relu -> fc(10)
        self.fc1 = nn.Linear(self.flatten_dim, fc1_width)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_width, fc2_width)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_width, class_count)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        logits = self.fc3(x)
        return logits
