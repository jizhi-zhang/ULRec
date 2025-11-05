from .movie import MovieENV, Movie_Grounding_Model
from .steam_no_same_item import Steam_Grounding_Model
from .amazon_no_same_item import Amazon_Grounding_Model
from .steam_no_same_item import SteamENV_no_same_item
from .amazon_no_same_item import AmazonENV_no_same_item

def get_envs(name, config, split):
    if name == "movie":
        return MovieENV(config)
    elif name == "steam_no_same_item":
        return SteamENV_no_same_item(config, split)
    elif name == "amazon_no_same_item":
        return AmazonENV_no_same_item(config, split)
    else:
        raise ValueError("Unknown env name: {}".format(name))
    
def get_groundingmodel(name, path, config, split):
    if name == "movie":
        return Movie_Grounding_Model(path, config)
    elif name == "steam":
        return Steam_Grounding_Model(path, config)
    elif name == "steam_no_same_item":
        return Steam_Grounding_Model(path, config)
    elif name == 'amazon':
        return Amazon_Grounding_Model(path, config)
    elif name == 'amazon_no_same_item':
        return Amazon_Grounding_Model(path, config)
    else:
        raise ValueError("Unknown env name: {}".format(name))