

import torch
import torchvision
import torchvision.transforms as transforms
#import Albumentation

import numpy as np
from albumentations import Compose, RandomCrop,PadIfNeeded, Normalize, HorizontalFlip,HueSaturationValue,Cutout,ShiftScaleRotate
from albumentations.pytorch import ToTensor
import cv2 


class album_Compose:
  def __init__(self):
    self.alb_transform =Compose([
      PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=False, p=.50),
      RandomCrop(32, 32, always_apply=False, p=.50),HorizontalFlip(), 
      HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
      Cutout(num_holes=1, max_h_size=8, max_w_size=8),
      ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2,rotate_limit=13, p=0.6),
      
      Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010],
        ),
      ToTensor(),
      ],p=.8)
  def __call__(self,img):
    img = np.array(img)
    img = self.alb_transform(image=img)['image']
    return img


def getData():
  k= album_Compose()
  transform_test = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  transform_train = transforms.Compose([  
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
      transforms.RandomRotation((-10.0, 10.0)), transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=k)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size= 500,
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                          shuffle=False, num_workers=2)
  testset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.Compose([transforms.ToTensor()]))
  testloader2 = torch.utils.data.DataLoader(testset2, batch_size=1,
                                          shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader, testloader,testloader2, classes



