import os
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from fastai.vision.all import *

def convert_mask_to_rgb(mask):
    class_to_rgb = {
             0: (0, 0, 0),
             1: (80,0,165),
             2: (255,204,0),
             3: (0,244,244),
             4: (105,105,105),    
             5: (255, 255, 102),
             6: (255, 153, 153),
             7: (127, 222, 209),
             8: (222, 127, 219),
             9: (108, 67, 107), 
             10: (152, 134, 94), 
             11: (99, 228, 150),
             12: (99, 176, 228), 
             13: (255, 186, 186), 
             14: (204, 0, 102), 
             15: (102, 51, 0), 
             16: (204, 204, 255),
             17: (0, 102, 204), 
             18: (255, 0, 255), 
             19: (102, 0, 204), 
             20: (141, 51, 117), 
             21: (145, 187, 192), 
             22: (155, 123, 219), 
             23: (193, 0, 76),
             24: (204, 255, 153)
    }


    red = np.zeros_like(mask)
    green = np.zeros_like(mask)
    blue = np.zeros_like(mask)
    
    num_classes = 25

    for class_idx in range(0, num_classes):
        red_temp = np.where(mask == class_idx, class_to_rgb[class_idx][0], 0)
        green_temp = np.where(mask == class_idx, class_to_rgb[class_idx][1], 0)
        blue_temp = np.where(mask == class_idx, class_to_rgb[class_idx][2], 0)

        red = np.where(red_temp > 0, red_temp, red)
        green = np.where(green_temp > 0, green_temp, green)
        blue = np.where(blue_temp > 0, blue_temp, blue)


    rgb = np.dstack((red, green, blue))
    return rgb

def convert_label_to_png(image_chip_label_path, image_chip_label_name, image_chip_name, data_png_path):
    label = rio.open(image_chip_label_path)
    band1 = label.read(1) 

    save_to_path = data_png_path + '/' + image_chip_name
    if os.path.isdir(save_to_path) == False:
        os.mkdir(save_to_path)

    save_to_path = save_to_path + '/labels/'
    if os.path.isdir(save_to_path) == False:
        os.mkdir(save_to_path)

    mask_rgb = convert_mask_to_rgb(band1)
    plt.imsave(save_to_path + 'labels' + '.png', mask_rgb.astype('uint8'))


def convert_all_labels_to_png(all_labels_path):
    # if folder for data_png doesn't exist, create one
    data_png_path = '../data_png'

    if os.path.isdir(data_png_path) == False:
        os.mkdir(data_png_path)

    for image_chip_path in Path(all_labels_path).ls():
        image_chip_label_name = str(image_chip_path)[-36:]
        print(image_chip_label_name)
        image_chip_name = image_chip_label_name[:-14] + image_chip_label_name[-7:]
        print(image_chip_name)
        print(image_chip_name + '/labels.tif')

        image_chip_label_path = all_labels_path + '/' + image_chip_label_name + '/labels.tif'
        convert_label_to_png(image_chip_label_path, image_chip_label_name, image_chip_name, data_png_path)


def convert_source_to_png(image_chip_source_path, image_chip_source_name, image_chip_name, data_png_path):
    source = rio.open(image_chip_source_path)
    red = source.read(4)
    green = source.read(3)
    blue = source.read(2)

    rgb = np.dstack((red, green, blue))
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(int)

    save_to_path = data_png_path + '/' + image_chip_name
    if os.path.isdir(save_to_path) == False:
        os.mkdir(save_to_path)

    save_to_path = save_to_path + '/source'
    if os.path.isdir(save_to_path) == False:
        os.mkdir(save_to_path)

    save_to_path = save_to_path + '/' + image_chip_source_name + '/'
    if os.path.isdir(save_to_path) == False:
        os.mkdir(save_to_path)

    plt.imsave(save_to_path + 'source' + '.png', rgb.astype('uint8'))


def check_image_with_no_clouds(image_chip_cloudmask_path):
    label = rio.open(image_chip_cloudmask_path)
    band1 = label.read(1) 
    non_zero_pixels = np.count_nonzero(band1)
    print(non_zero_pixels)
    max_pixels_with_clouds = 100

    if (non_zero_pixels < max_pixels_with_clouds):
        return True 
    else:
        return False 



def convert_all_sources_to_png(all_sources_path):
    data_png_path = '../data_png'

    if os.path.isdir(data_png_path) == False:
        os.mkdir(data_png_path)
    
    for image_chip_path in Path(all_sources_path).ls():
        print(image_chip_path)
        image_chip_source_name = str(image_chip_path)[-50:]
        print(image_chip_source_name)

        image_chip_name = image_chip_source_name[:-11]
        print(image_chip_name)
        image_chip_name = image_chip_name[:-17] + image_chip_name[-7:]
        print(image_chip_name)

        image_chip_source_path = all_sources_path + '/' + image_chip_source_name + '/source.tif'
        print(image_chip_source_path)

        image_chip_cloudmask_path = all_sources_path + '/' + image_chip_source_name + '/cloudmask.tif'
        image_with_no_clouds = check_image_with_no_clouds(image_chip_cloudmask_path)
        print(image_chip_cloudmask_path)

        if (image_with_no_clouds):
            convert_source_to_png(image_chip_source_path, image_chip_source_name, image_chip_name, data_png_path)