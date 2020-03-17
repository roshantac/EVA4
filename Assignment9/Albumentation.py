import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip,HueSaturationValue,Cutout,ShiftScaleRotate
from albumentations.pytorch import ToTensor


class album_Compose:
  def __init__(self):
    self.alb_transform =Compose([

      HorizontalFlip(),
      HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
      Cutout(num_holes=1, max_h_size=8, max_w_size=8),
      ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2,rotate_limit=13, p=0.8),
      
      Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010],
        ),
      ToTensor(),
      ],p=.5)
  def __call__(self,img):
    img = np.array(img)
    img = self.alb_transform(image=img)['image']
    return img
