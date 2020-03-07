import torch
import torchvision
import torchvision.transforms as transforms
from dataset import *
trainloader, testloader, classes = getData()

import torch.nn as nn
import torch.nn.functional as F
from base_functions import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()       
        self.model = nn.Sequential(Conv2d_BN(3,32,dropout=0.1,dilate=2),   # 28
                                Conv2d_BN(32,48,dropout=0.1),  # 26 #
                                Maxpooling(2),                 # 13
                                DepthwiseConv2D(48,96,1,dropout=0.1),
                                Conv2d_BN(96,128,dropout=0.1), # 9
                                DepthwiseConv2D(128,256,1,dropout=0.1),
                                Conv2d(256,328),# 5
                                gap(5),
                                Conv1x1(328,10),                                
                                )
    
    def forward(self,x):
      x = self.model(x)
      x=x.view(-1,10)
      return F.log_softmax(x, dim= -1)
    

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(3, 32, 32))

from traing import *
Training(15,model,device, trainloader, testloader )

ClassTestAccuracy(testloader,device,model,classes)