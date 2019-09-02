import os
import pickle


MODEL_PATH = "models/pickles/"
os.makedirs(MODEL_PATH, exist_ok=True)

def parse_hyper_parameters(args):
    hyper_parameters = {}
    for arg in args:
        name, value = arg.split("=", maxsplit=1)
        hyper_parameters[name] = value
    return hyper_parameters

def model_name(module_name, hyper_parameters={}):
    suffix = "-".join(f"{k}{v}" for k, v in hyper_parameters.items())
    return f"{module_name}-{suffix}"

def model_path(model_name):
    return os.path.join(MODEL_PATH, model_name+".pickle")

def save(model, module_name, hyper_parameters={}):
    name = model_name(module_name, hyper_parameters)
    path = model_path(name)
    with open(path, 'wb') as file:
        print("Saving model to", path)
        pickle.dump(model, file)
        print("Done!")

def load(module_name, hyper_parameters={}):
    name = model_name(module_name, hyper_parameters)
    path = model_path(name)
    with open(path, 'rb') as file:
        print("Loading model from", path)
        model = pickle.load(file)
        print("Loaded!")
        return model
