from collections import Counter

from tqdm import tqdm

from models.baseline.simple_ngram import Model as BaseModel

PARAM_GRID = [
    {'n': 2, 'level': "byte", 'norm': "False", 'L': "200"},
    {'n': 3, 'level': "char", 'norm': "False", 'L': "300"},
    {'n': 4, 'level': "char", 'norm': "False", 'L': "400"},
    {'n': 5, 'level': "char", 'norm': "False", 'L': "400"},
    {'n': 6, 'level': "char", 'norm': "False", 'L': "400"},
    {'n': 2, 'level': "word", 'norm': "True",  'L': "200"},
    {'n': 2, 'level': "flex", 'norm': "True",  'L': "150"}
]
                

class Model:
    def __init__(self, data):
        data = list(data) # strip iterators, sorry tqdm!
        self.ntweets = Counter(t.handle for t in tqdm(data))
        self.models = [BaseModel(tqdm(data), **params) for params in tqdm(PARAM_GRID)]
    def predict(self, tweet):
        votes = Counter(model.predict(tweet) for model in self.models)
        majority = max(votes.keys(), key=lambda a: (votes[a], self.ntweets[a]))
        return majority
