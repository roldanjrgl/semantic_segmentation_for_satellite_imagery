import numpy as np
from fastai.vision.all import *
import rasterio as rio
import os 
import matplotlib.pyplot as plt

def convert_mask_to_rgb(mask, output_masks_rgb):
    class_to_rgb = {
             1: (80,0,165),
             2: (255,204,0),
             3: (0,244,244),
             4: (105,105,105)}    

    red = np.zeros_like(mask)
    green = np.zeros_like(mask)
    blue = np.zeros_like(mask)
    
    num_classes = 4
    for class_idx in range(1, num_classes + 1):
        red_temp = np.where(mask == class_idx, class_to_rgb[class_idx][0], 0)
        green_temp = np.where(mask == class_idx, class_to_rgb[class_idx][1], 0)
        blue_temp = np.where(mask == class_idx, class_to_rgb[class_idx][2], 0)

        red = np.where(red_temp > 0, red_temp, red)
        green = np.where(green_temp > 0, green_temp, green)
        blue = np.where(blue_temp > 0, blue_temp, blue)


    rgb = np.dstack((red, green, blue))
    return rgb

def convert_all_masks_to_png(all_masks_path, output_masks_rgb):
    if os.path.isdir(output_masks_rgb) == False:
        os.mkdir(output_masks_rgb)

    for image_chip_path in Path(all_masks_path).ls():
        file_type = str(image_chip_path)[-6:]
        if (file_type == '_m.png'):
            print(image_chip_path)
            print(str(image_chip_path).split('/')[-1])
            image_chip_name = str(image_chip_path).split('/')[-1]


            mask = rio.open(str(image_chip_path)).read(1)
            mask_rgb = convert_mask_to_rgb(mask, output_masks_rgb)
            plt.imsave(output_masks_rgb + '/' + image_chip_name, mask_rgb.astype('uint8'))


def main():
    all_masks_path = '../../../../data_sources/landcover.ai.v1/output'
    output_masks_rgb = 'masks_rgb'
    convert_all_masks_to_png(all_masks_path, output_masks_rgb)


if __name__ == "__main__":
    main()