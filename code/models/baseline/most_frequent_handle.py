import numpy as np

def train(data):
    handles = [t.handle for t in data]
    model = MostFrequentModel(handles)
    return model

class MostFrequentModel:
    def __init__(self, handles):
        handles = np.array([int(handle) for handle in handles])
        self.mostFreq = str(np.bincount(handles).argmax())
    def predict(self, tweet):
        return self.mostFreq