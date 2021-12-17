import albumentations as A
import numpy as np
import os, re, sys, random, shutil, cv2

def get_class_names():
    return ['unlabeled', 'building', 'woodland', 'water', 'roads']

def get_class_to_rgb_map():
    class_to_rgb_map = {}
    class_to_rgb_map['unlabeled'] = np.array([0,0,0])
    class_to_rgb_map['building'] = np.array([80,0,165])
    class_to_rgb_map['woodland'] = np.array([255,204,0])
    class_to_rgb_map['water'] = np.array([0,244,244])
    class_to_rgb_map['roads'] = np.array([105,105,105])
    return class_to_rgb_map

def get_data_info(name='landcoverai', path=None):
    return DataInfo(name, path)

class DataInfo():
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.class_names = get_class_names()
        self.class_to_rgb_map = get_class_to_rgb_map()
        self.height = 512
        self.width = 512

        self.class_codes = []
        for cname in self.class_names:
            rgb_arr = self.class_to_rgb_map[cname]
            self.class_codes.append(tuple([rgb_arr[0], rgb_arr[1], rgb_arr[2]]))

        self.imap = {k:v for k,v in enumerate(self.class_codes)}
        self.nimap = {v:k for k,v in enumerate(self.class_names)}
        self.inmap = {k:v for k,v in enumerate(self.class_names)}

    def __augment__(self):
        transform = A.Compose([
            A.RandomCrop(width=self.width, height=self.height, p=1.0),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
            A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
            A.OneOf([
                A.CLAHE (clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
            ], p=1.0),
        ], p=1.0)
    
        return transform