from models.POLYCO import POLYCO
from models.POLYCORE import POLYCORE

def load_model(config):

    if config['MODEL'] in 'POLYCO':
        model = POLYCO(config)
    elif config['MODEL'] == 'POLYCORE':
        model = POLYCORE(config)
    else:
        ValueError('No model selected')

    return model