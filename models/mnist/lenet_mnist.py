from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pdb import set_trace as bp

class lenet_mnist(nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.cuda()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#self.reshape = 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(980, 50)
        self.fc2 = nn.Linear(50, 10)

#def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 980)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x, dim=1)
    def forward(self,x):
        #x = x.to(torch.device("cuda:0"))
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.conv2_drop(out)
        out = self.maxpool2(out)
        out = self.relu2(out)
        out = out.view(-1,980)
        out = self.fc1(out)
        #bp()
        #out = nn.Linear(980,50).cuda()(out)
        #out = F.relu(out)
        out = self.fc2(out)
        #out = nn.Softmax(out)
        return out #nn.Softmax(out)
def lenet_mnist_28():
    return lenet_mnist()

