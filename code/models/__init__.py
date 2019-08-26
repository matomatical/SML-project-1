import os
import pickle

MODEL_PATH = "models/pickles/"

def model_path(name):
    return os.path.join(MODEL_PATH, name+".pickle")

def save(model, name):
    with open(model_path(name), 'wb') as file:
        pickle.dump(model, file)

def load(name):
    with open(model_path(name), 'rb') as file:
        return pickle.load(file)
