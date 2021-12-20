# Semantic Segmentation for Satellite Imagery

The package data_preprocessing contains all the scripts we used to collect raw datasets from landcoverai, landcovernet, and ghanacropcover datasets. It also contains the code to clean up the images and generate RGB masks for each source.

Our entire training and testing pipeline is governed by a configuration file provided as an input. The configuration file contains info regarding the model, weight initialization, hyperparameters, and the path to dataset that we want to train/test.

    1. config.json files contain all the configuration required for a particular run.
    2. driver.py is the entry point for our training/testing. The script runs in two modes - "train" and "test".
    3. Command to start training/test = "python driver.py mode <path-to-config-file>"
    
Breakdown of a config file

    1. network - model that we want to train on. Either one of "unet" or "unet++".
    2. encoder_name - backbone of the model. Examples include - "resnet18", "resnet34", "resnet50", "resnet101".
    3. encoder_weights - weights used to initialize the encoder.
    4. dataset_name - dataset that we want to work on. We ran experiments on "landcoverai", "landcovernet", and "ghanacropcover".
    5. dataset_path -local fs path where the data is stored.
    6. epochs - duration of the training.
    7. batch_size - batch size of training.
    8. activation_fn - defines output of our network node.
    9. loss_fn - loss function to dictate training. either one of "focal_loss", "cross_entropy_loss", and "dice_loss".
    10. lr - learning rate
    11. device - either one of cuda/cpu
    12. model_out_path - path where we want to save the model (training) or where we want to load the model from (testing)
    13. log_out_path - path to store tensorboard logs.
