import json
import numpy as np
import pickle
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class JsonHelper:
    def save_to_json(self, name, object):
        with open(name + '.json', 'w') as handle:
            json.dump(object, handle, cls=NumpyEncoder)

    def load_from_json(self, name):
        with open(name + '.json') as handle:
            data = json.load(handle)
        return data

    def pickle_object(self, name, object):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)

    def load_pickle_object(self, name):
        with open(name + '.pickle', 'rb') as handle:
            return pickle.load(handle)


