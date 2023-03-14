import json
import numpy as np

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