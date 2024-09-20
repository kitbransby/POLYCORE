import sys
sys.path.append('..')
import argparse
import datetime
import yaml
import torch
from utils.IVUSDataset import load_dataset
from POLYCO_Trainer import Trainer as POLYCO_Trainer
from POLYCORE_Trainer import Trainer as POLYCORE_Trainer
from models.load_model import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DIR', type=str, default='../')
    parser.add_argument('--BATCH_SIZE', type=int)
    parser.add_argument('--VAL_BATCH_SIZE', type=int)
    parser.add_argument('--PREPROCESSING', type=str, default='AR')
    parser.add_argument('--CFG', type=str, default='')
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('../config/'+cmd_config['CFG']+'.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config) # command line args overide yaml

    config['NAME'] = datetime.datetime.now().strftime('%m_%d_%H_%M_%S.%f') + '_' + config['MODEL']
    config['N_CLASSES'] = 3
    config['MASK_FORMAT'] = '0-1-2'
    config['ACTIVATION'] = None
    config['IMAGE_INP'] = 'Image_Affine_480'
    config['IMAGE_DIM'] = 3
    config['DEVICE'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Connected to: ', config['DEVICE'])

    print('Config: ', config)

    if config['TRAINER'] == 'POLYCO': # POLYCO TRAINER (BASE MODEL)
        train_dataset, test_dataset = load_dataset(config)
        model = load_model(config)
        POLYCO_Trainer(train_dataset, test_dataset, model, config)
    elif config['TRAINER'] == 'POLYCORE': # DECOUPLE DISTANCE AND DIRECTION TO TWO MAPS.
        train_dataset, test_dataset = load_dataset(config)
        model = load_model(config)
        POLYCORE_Trainer(train_dataset, test_dataset, model, config)
    else:
        print('No Trainer selected')
