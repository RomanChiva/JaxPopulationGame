from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.LR = 0.01          # Alpha
    config.DISCOUNT = 0.99
    config.DELTA_WIN = 0.005  # Slow learning when winning
    config.DELTA_LOSE = 0.02  # Fast learning when losing
    config.Q_INIT_MEAN = 0.0
    config.Q_INIT_STD = 0.01
    
    return config
