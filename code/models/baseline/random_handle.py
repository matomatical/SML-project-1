import numpy as np


class Model:
    def __init__(self, data):
        self.handles = np.array([int(t.handle) for t in data])
        self.rng = np.random.default_rng()
    def predict(self, tweet):
        return str(self.rng.choice(self.handles))

