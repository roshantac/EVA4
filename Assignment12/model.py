
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


def ResBlock(channel_no): # as per rohans question
  return nn.Sequential(
  nn.Conv2d(in_channels = channel_no, out_channels = channel_no, kernel_size = (3,3), padding=1, dilation =1),
  nn.BatchNorm2d(channel_no),
  nn.ReLU(),   
  nn.Conv2d(in_channels = channel_no, out_channels = channel_no, kernel_size = (3,3), padding=1, dilation =1),
  nn.BatchNorm2d(channel_no),
  nn.ReLU()
  )

def ConvI(inChannels, outChannels,kernel=3, padding=1, b_bias=False):
  return nn.Sequential(
      nn.Conv2d(in_channels=inChannels, out_channels=outChannels, 
                kernel_size=(kernel, kernel), padding=padding, bias=b_bias),            
      nn.BatchNorm2d(outChannels),
      nn.ReLU(),
  )
def ConvII(inChannels, outChannels,kernel=3, padding=1):
  return nn.Sequential(
      nn.Conv2d(in_channels=inChannels, out_channels=outChannels, 
                kernel_size=(kernel, kernel), padding=padding),
      nn.MaxPool2d(2, 2),            
      nn.BatchNorm2d(outChannels),
      nn.ReLU(),
  )
def Maxpooling(kernel):
  return nn.MaxPool2d(kernel, kernel)



class Model11(nn.Module):
  def __init__(self):
      super(Model11, self).__init__()
      self.Preplayer = ConvI(3, 64) 
      #layer 1
      self.ConvPool1 = ConvII(64, 128) 
      self.Res1 = ResBlock(128) 
      #layer 2
      self.ConvPool2 = ConvII(128, 256)
      #layer 3
      self.ConvPool3 = ConvII(256, 512)
      self.Res2 = ResBlock(512)
      #layer 4
      self.Pool = Maxpooling(4)
      self.FC   = nn.Linear(512, 10)


  def forward(self,x):
    X = self.Preplayer(x) #32
    X = self.ConvPool1(X) 
    R1 = self.Res1(X)     #16
    X = self.ConvPool2(X + R1) 
    X = self.ConvPool3(X)
    R2 = self.Res2(X)
    X = self.Pool(X+R2 )
    X = X.view(-1,512)
    X = self.FC(X)
    return F.log_softmax(X, dim= -1)


