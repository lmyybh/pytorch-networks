import yaml

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def yaml2obj(filename):
    with open(filename, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    return dict2obj(data)
