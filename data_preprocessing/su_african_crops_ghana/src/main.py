import os
from matplotlib.colors import Colormap
import rasterio as rio
from fastai.vision.all import *
from preprocessing import *

def main():
    all_labels_path = '../su_african_crops_ghana_labels'
    all_source_path = '../su_african_crops_ghana_source_s2'

    convert_all_labels_to_png(all_labels_path)
    convert_all_sources_to_png(all_source_path)


if __name__ == "__main__":
    main()