# Semantic Segmentation for Satellite Imagery
Steps to Train/Test

Our entire training and testing pipeline is governed by a configuration file provided as an input. The configuration file contains info regarding the model, weight initialization, hyperparameters, and the path to dataset that we want to train/test.

    1. config.json files contain all the configuration required for a particular run.
    2. driver.py is the entry point for our training/testing. The script runs in two modes - "train" and "test".
    3. Command to start training/test = "python driver.py mode <path-to-config-file>"
