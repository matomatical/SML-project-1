import numpy as np


class Model:
    def __init__(self, data):
        handles = np.array([int(t.handle) for t in data])
        self.mostFreq = str(np.bincount(handles).argmax())
    def predict(self, tweet):
        return self.mostFreq