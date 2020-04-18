import time
import numpy as np
import os
import zipfile
import requests 
import io
#from io import StringIO, BytesIO
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch
from io import BytesIO

path = 'tiny-imagenet-200'
def downloadDataset():
    if(os.path.isdir("tiny-imagenet-200")):
        print(" Already downloaded")
        return
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    r = requests.get(url,stream=True)
    print("downloads completed")
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall('./')
    print("Extraction completed")
    zip_ref.close()


def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open(path + '/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

def get_class_to_id_dict():
    id_dict = self.get_id_dictionary()
    all_classes = {} # contains class and ids
    class_labels = [] # list of clasess
    result = {}
    for i, line in enumerate(open(path + '/words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
        class_labels.append(word)
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key]) #
    print( "result : ", result)
    print("clss labels", class_labels)
    return result, class_labels

def get_train_test_data(ratio=0.3):
    print('Starting data loading')
    id_dict = get_id_dictionary()
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    total_val_images = 10000
    images_for_class = 500
    #Creating Train Class
    train_image_count = images_for_class - (images_for_class * ratio)
    for key, value in id_dict.items():
        all_data, all_labels = [], []
        for i in range(images_for_class):
            all_data.append(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)))
            all_labels.append(id_dict[key])

        for x in range(0, images_for_class):
            if x < train_image_count:
                train_data.append(all_data[x])
                train_labels.append(all_labels[x])
            else:
                test_data.append(all_data[x])
                test_labels.append(all_labels[x])
    # Creating Test Class
    test_image_count = total_val_images - (total_val_images * ratio)
    count = 0
    for line in open(path + '/val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        if count < test_image_count:
            train_data.append(path + '/val/images/{}'.format(img_name))
            train_labels.append(id_dict[class_id])
        else:
            test_data.append(path + '/val/images/{}'.format(img_name))
            test_labels.append(id_dict[class_id])
        count += 1

    print('Finished data loading, in {} seconds'.format(time.time() - t))
    return np.array(train_data), train_labels, np.array(test_data), test_labels

def GetClasses():
    id_dict = get_id_dictionary()
    _, class_labels = get_class_to_id_dict()
    return class_labels, id_dict



class TinyImagenetDataset(Dataset):

    def __init__(self, image_data, image_labels, transform=None):
        self.transform = transform
        self.image_data = image_data
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_data[idx]))
        label = self.image_labels[idx]
        #if image is grey colour convert into 3 channels
        if (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1):
            image = np.stack((image,) * 3, axis=-1)

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label


def getData():
 # print("Downloading Dtataset, Please wait...")
	downloadDataset()
	trnData, trnLbl, tstData, tstLbl = get_train_test_data()
	from dataset import album_Compose
	trnTransform = album_Compose()
	trainSet = TinyImagenetDataset(trnData, trnLbl, transform = trnTransform)
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size= 500,
                                            shuffle=True, num_workers=2)
	tstTransform = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize((0.48043722, 0.44820285, 0.39760238), (0.27698976, 0.26908714, 0.2821603))])
	testSet = TinyImagenetDataset(tstData, tstLbl, transform = tstTransform)
	testLoader = torch.utils.data.DataLoader(testSet, batch_size= 500,
                                            shuffle=True, num_workers=2)

	return trainLoader, testLoader