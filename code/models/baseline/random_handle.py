import numpy as np

def train(data):
    handles = [t.handle for t in data]
    model = RandomHandleModel(handles)
    return model

class RandomHandleModel:
    def __init__(self, handles):
        self.rng = np.random.default_rng()
        self.handles = np.array([int(handle) for handle in handles])
    def predict(self, tweet):
        return str(self.rng.choice(self.handles))

