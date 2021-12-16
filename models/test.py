_pathfrom data.datasets import *
from data.dataloader import *
from helpers.utils import *

import segmentation_models_pytorch as smp
import sys

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

def test(network='unet', encoder='resnet34', encoder_weights='imagenet', dataset_name='landcoverai', dataset_path=None, epochs=40, batch=32, act='softmax', loss_name='focal_loss', lr=0.00001, device_name='cuda', model_path=None, tb_writer_path=None):
    ds_info = get_data_info(dataset_name, dataset_path)
    device = torch.device(device_name)
    test_loader = load_data(ds_info, "test", batch_size=batch)

    if network == "unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(ds_info.class_names), activation=act)
    elif network == "unet++":
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(ds_info.class_names), activation=act)
    else:
        raise ValueError("Unrecognized Network Name")
    loss = get_loss_function(loss_name)

    metrics = get_metrics_to_capture()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_epoch = smp.utils.train.ValidEpoch(model,loss=loss,metrics=metrics,device=device,verbose=True)




