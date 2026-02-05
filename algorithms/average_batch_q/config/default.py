from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.LR = 0.01
    config.DISCOUNT = 0.99
    config.EPSILON = 0.1
    config.Q_INIT_MEAN = 0.0
    config.Q_INIT_STD = 0.01
    config.BATCH_SIZE = 32 # If we use a fixed batch size
    
    return config
