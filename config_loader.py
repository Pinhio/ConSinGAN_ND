import json

class Config:
    '''
    Simple config class for converting a JSON to a class.
    This makes access of attributes easier.
    
    Parameters
    ----------
    config : dict
        the JSONs content
    '''
    def __init__(self, config:dict) -> None:
        for k, v in config.items():
            setattr(self, k, v)

def load_config() -> dict:
    '''
    Loads the config.json file and returns an instance of Config.
    '''
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    return Config(cfg)