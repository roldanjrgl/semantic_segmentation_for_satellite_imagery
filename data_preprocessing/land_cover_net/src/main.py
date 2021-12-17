import os
from data_preprocessing import *

def main():
    all_source_path = 'data_all_source'
    all_labels_path = './data_all_labels/ref_landcovernet_v1_labels'

    convert_all_sources_to_png(all_source_path)
    convert_all_labels_to_png(all_labels_path)
        
if __name__ == "__main__":
    main()