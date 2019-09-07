from collections import Counter

from models.baseline.simple_ngram import Model as BaseModel

PARAM_GRID = [{'n': 2, 'level': "byte", 'norm': "False", 'L': "200"},
              {'n': 3, 'level': "char", 'norm': "False", 'L': "300"},
              {'n': 4, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 5, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 6, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 2, 'level': "word", 'norm': "True",  'L': "200"}]
                

class Model:
    def __init__(self, data):
        data = list(data) # strip iterators, sorry tqdm!
        self.models = [BaseModel(data, **params) for params in PARAM_GRID]
    def predict(self, tweet):
        return majority(model.predict(tweet) for model in self.models)

def majority(items):
    return Counter(items).most_common(1)[0][0]
        
