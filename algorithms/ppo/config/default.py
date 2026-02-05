from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    # Network
    config.ACTIVATION = "tanh"
    
    # PPO Hyperparameters
    config.LR = 0.01
    config.NUM_STEPS = 128
    config.NUM_MINIBATCHES = 4
    config.UPDATE_EPOCHS = 1
    config.GAMMA = 0.99
    config.GAE_LAMBDA = 0.95
    config.CLIP_EPS = 0.2
    config.ENT_COEF = 0.01
    config.VF_COEF = 0.5
    config.MAX_GRAD_NORM = 0.5
    
    return config
