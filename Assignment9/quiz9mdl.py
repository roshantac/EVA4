import torch.nn as nn
import torch.nn.functional as F
from base_functions import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()       
        self.Conv1 = Conv2d_BN(3,3,dropout=0.1,padding=1)
        self.pool = Maxpooling(2)
        self.Conv2 = Conv2d_BN(3,32,dropout=0.1,padding=1)
        self.Conv3 = Conv2d_BN(32,64,dropout=0,padding=1)
        self.GAP_1 = gap(8)
        self.convF = Conv1x1(64,10)
    
    def forward(self,x1):
      x2 = self.Conv1(x1)
      x3 = self.Conv1(x1 + x2)
      x4 = self.pool(x1 + x2 + x3)
      x5 = self.Conv2(x4)
      x6 = self.Conv2(x4 + x5)
      x7 = self.Conv2(x4 + x5 + x6)
      x8 = self.pool(x5 + x6 + x7)
      x9 = self.Conv3(x8)
      x10 = self.Conv3(x8 + x9)
      x11 = self.Conv3(x8 + x9 + x10)
      x12 = self.GAP_1(x11)
      x13 = self.convF(x12)
      x14 = x13.view(-1,10)
      return F.log_softmax(x14, dim= -1)
