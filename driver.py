import sys
import json
from models.train import *
from models.test import *

def main():
    config_path = sys.argv[1]
    mode = sys.argv[2]
    f = open(config_path)
    config = json.load(f)
    print("configs are loaded")
    if mode == "test":
        test(network=config['network'], encoder=config['encoder_name'], encoder_weights=config['encoder_weights'], dataset_name=config['dataset_name'], dataset_path=config['dataset_path'], epochs=config['epochs'], batch=config['batch_size'], act=config['activation_fn'], loss_name=config['loss_fn'], lr=config['lr'], device_name=config['device'], model_path=config['model_out_path'], tb_writer_path=config['log_out_path'])
    else:
        train_and_validate(network=config['network'], encoder=config['encoder_name'], encoder_weights=config['encoder_weights'], dataset_name=config['dataset_name'], dataset_path=config['dataset_path'], epochs=config['epochs'], batch=config['batch_size'], act=config['activation_fn'],    loss_name=config['loss_fn'], lr=config['lr'], device=config['device'], model_path=config['model_out_path'], tb_writer_path=config['log_out_path'])

if __name__ == "__main__":
    print("firing up the thrusters")
    main()
