import torch
import torch.nn as nn
import sys
import os
from copy import deepcopy
from .quantization import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spikingjelly')))

from spikingjelly.activation_based import layer, neuron

__all__ = [
    'cifar10net'
]



class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, **kwargs):
        super().__init__()

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv2d = quan_Conv2d(in_channels, channels, kernel_size=3,  padding=1, bias=False)
                bn = layer.BatchNorm2d(channels)
                node = neuron.ParametricQuanLIFNode(**deepcopy(kwargs))
                node.pre_layer = conv2d

                conv.append(conv2d)
                conv.append(bn)
                conv.append(node)

            conv.append(layer.MaxPool2d(2, 2))

        liner1 = quan_Linear(channels * 8 * 8, 2048)
        node1 = neuron.ParametricQuanLIFNode(**deepcopy(kwargs))
        node1.pre_layer = liner1
        
        liner2 = quan_Linear(2048, 100)
        node2 = neuron.ParametricQuanLIFNode(**deepcopy(kwargs))
        node2.pre_layer = liner2


        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            liner1,
            node1,

            layer.Dropout(0.5),
            liner2,
            node2,

            layer.VotingLayer(10)
        )

    def forward(self, x):
        return self.conv_fc(x)
    
    
def cifar10net(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CIFAR10Net()
    return model