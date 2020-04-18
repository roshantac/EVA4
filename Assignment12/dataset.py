

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from albumentations import Compose, RandomCrop,PadIfNeeded, Normalize, HorizontalFlip,HueSaturationValue,Cutout,ShiftScaleRotate
from albumentations.pytorch import ToTensor
import cv2 


class album_Compose:
  def __init__(self):
    self.alb_transform =Compose([
      PadIfNeeded(min_height=64, min_width=64, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=False, p=.50),
      RandomCrop(64, 64, always_apply=False, p=.50),
      HorizontalFlip(), 
      HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
      Cutout(num_holes=1, max_h_size=8, max_w_size=8),
      ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2,rotate_limit=13, p=0.6),
      
      Normalize(
        mean = [0.48043722, 0.44820285, 0.39760238],
        std = [0.27698976, 0.26908714, 0.2821603],
        ),
      ToTensor(),
      ],p=.8)
  def __call__(self,img):
    img = np.array(img)
    img = self.alb_transform(image=img)['image']
    return img



