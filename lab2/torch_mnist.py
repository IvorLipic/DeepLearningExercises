import torch
import numpy as np
from torch import nn
from scipy.stats import truncnorm

class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels=1, conv1_width=16, conv2_width=32,
                 fc1_width=512, class_count=10):
        super(ConvolutionalModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5,
                               stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5,
                               stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        # Calculating the flatten dimension.
        # Input 1x28x28 -> conv1 -> 16x28x28    
        #               -> pool1 -> 16x14x14 -> relu
        #               -> conv2 -> 32x14x14
        #               -> pool2 -> 32x7x7
        # so:
        # flatten_dim = conv2_width(32) * 47 * 7
        self.flatten_dim = conv2_width * 7 * 7

        self.fc1 = nn.Linear(self.flatten_dim, fc1_width)
        self.relu3 = nn.ReLU()
        self.fc_logits = nn.Linear(fc1_width, class_count)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            self.fc_logits.reset_parameters()
        '''
        def variance_scaling_initializer(tensor, shape, fan_in, factor=2.0):
            #print(shape, fan_in)
            sigma = np.sqrt(factor / fan_in)
            weights = truncnorm(-2, 2, loc=0, scale=sigma).rvs(shape)
            return tensor.data.copy_(torch.from_numpy(weights).float())
        
        layers = [m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

        for m in layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                else:
                    fan_in = m.in_features

                with torch.no_grad():
                    m.weight.data = variance_scaling_initializer(m.weight.data,
                                                                 m.weight.data.shape,
                                                                 fan_in)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        '''

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.pool2(h)
        h = self.relu2(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.relu3(h)
        logits = self.fc_logits(h)
        return logits
