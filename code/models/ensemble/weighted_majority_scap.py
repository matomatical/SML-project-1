from collections import Counter, defaultdict

import numpy as np

from models.baseline.simple_ngram import Model as BaseModel

PARAM_GRID = [{'n': 2, 'level': "byte", 'norm': "False", 'L': "200"},
              {'n': 3, 'level': "char", 'norm': "False", 'L': "300"},
              {'n': 4, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 5, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 6, 'level': "char", 'norm': "False", 'L': "400"},
              {'n': 2, 'level': "word", 'norm': "True",  'L': "200"}]
                
K = 10
WEIGHTS = np.array([1,3,3,3,3,1]) # one weight for each model

class Model:
    def __init__(self, data):
        self.M = len(PARAM_GRID) # number of models
        data = list(data) # strip iterators, sorry tqdm!
        self.models = [BaseModel(data, **params) for params in PARAM_GRID]
    def predict(self, tweet):
        # list of top k predictions from each of the M models
        # each list is of the form [(handle 1, score 1), ..., (handle k, score k)]
        predictions = [model.predict(tweet, topk=K, scores=True) for model in self.models]

        # we want to collect the scores for each handle that appears in any
        # prediction list. let's create one vector for each such handle.
        # default the elements to 0 (so that if the handle wasn't in the top k for
        # any particular model, it effectively gets a score of 0 from that model)
        handlescores = {handle: np.zeros(self.M) for ps in predictions for handle, _score in ps}
        # then loop through again to fill in the values with non-zero scores
        for m, ps in enumerate(predictions):
            for handle, score in ps:
                handlescores[handle][m] = score / ps[0][1]
        # okay, now we are looking for argmax_handle (weights . handlescores[handle]):
        return max(handlescores, key=lambda h: WEIGHTS.dot(handlescores[h]))
