
import PIL
import numpy as np
from torchvision import transforms
from utils import visualize_cam
from gradCam import GradCAM
from Resnet import *

def show_map():
    target_model = ResNet18()
    gradcam = GradCAM.from_config(model_type='resnet', arch=target_model, layer_name='layer4')
    img=PIL.Image.open('test.jpg')
    img =  transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)[None]
    mask, logit = gradcam(img)#class_idx=10
    heatmap, cam_result = visualize_cam(mask, img)
    return heatmap, cam_result
   


