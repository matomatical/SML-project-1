import numpy as np
from collections import defaultdict

def train(data):
    model = SimpleNGramModel(data)
    return model

class SimpleNGramModel:
    def __init__(self, data):
        self.handles = [t.handle for t in data]
        self.text = [t.text for t in data]

        self.ngrams = defaultdict(defaultdict(int)) # {handle: {ngram: count, ...}, ...}

        for t in data:

            self.ngrams[t.handle][t.]

    # don't include start and end characters (for simplicity)
    def generateNgrams(self, text, n):
        # deal with n > len(text)?
        ngrams = []
        for i in range(len(text)-n):
            ngram = ""
            for j in range(n):
                pass
                