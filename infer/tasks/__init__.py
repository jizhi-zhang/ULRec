from .steam import SteamTask
from .amazon import AmazonTask

def get_task(name, split):
    if name == "steam":
        return SteamTask(split)
    elif name == "amazon":
        return AmazonTask(split)
    else:
        raise ValueError("Unknown task name: {}".format(name))
    
