from data.datasets import *
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

def train_and_validate(network='unet', encoder='resnet34', encoder_weights='imagenet', dataset_name='landcoverai', dataset_path=None, epochs=40, batch=32, act='softmax', loss_name='focal_loss', lr=0.00001, device='cuda', model_path=None, tb_writer_path=None):
    ds_info = get_data_info(dataset_name, dataset_path)

    train_loader = load_data(ds_info, "train", batch_size=batch)
    val_loader = load_data(ds_info, "validate", batch_size=batch)    

    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(ds_info.class_names), activation=act)
    loss = get_loss_function(loss_name)

    metrics = get_metrics_to_capture()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr),])

    train_epoch = smp.utils.train.TrainEpoch(model,loss=loss, metrics=metrics, optimizer=optimizer, device=device,verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model,loss=loss,metrics=metrics,device=device,verbose=True)


    writer = SummaryWriter(tb_writer_path)

    print("ready to train")
    max_score = 0
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        writer.add_scalars('train', build_tensorboard_metrics(train_logs, loss_name), i)
        writer.add_scalars('validate', build_tensorboard_metrics(valid_logs, loss_name), i)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model.state_dict(), model_path)
            print('Model saved!')

        if i == 20:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')
        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-7
            print('Decrease decoder learning rate to 1e-7!')
        print("max iou score we got till now is ", max_score)
    print("training complete...")
