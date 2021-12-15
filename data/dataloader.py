import os, re, sys, random, shutil, cv2
import numpy as np
from PIL import ImageColor
from IPython.display import SVG
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from data.datasets import *
from helpers.utils import *


def load_data(data_info_obj, category, batch_size):
    path = data_info_obj.path + "/" + category
    dataset = SatDataSet(path, transform=None, dinfo=data_info_obj)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=6)
    return loader

class SatDataSet(Dataset):

    def __init__(self, base_path, transform=None, dinfo=None):
        self.img_dir = base_path + "/images/"
        self.msk_dir = base_path + "/masks/"
        self.items = self.make_dataset()
        self.transform = None
        self.dinfo = dinfo

    def __len__(self):
        return len(self.items)

    def make_dataset(self):
        data_list = sorted([f for f in os.listdir(self.msk_dir) if os.path.isfile(os.path.join(self.msk_dir, f))])
        items = []
        for it in data_list:
            item = (os.path.join(self.img_dir, it.replace('.png', '.jpg')), os.path.join(self.msk_dir, it))
            items.append(item)
        return items

    def __getitem__(self, idx):
        img_path, msk_path = self.items[idx]
        img = io.imread(img_path)
        msk = io.imread(msk_path)
        if self.transform is not None:
            img = self.transform(img)
            msk = self.transform(msk)
        mask_encoded = convert_to_onehot(msk, self.dinfo.imap)
        img = img.transpose(2, 1, 0).astype('float32') / 255
        mask_encoded = mask_encoded.transpose(2,1,0)
        return img, mask_encoded