import sys
import time
import importlib

from tqdm import tqdm

import data
import models

data.load_all_train()
data.load_test()

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = models.parse_hyper_parameters(sys.argv[2:])
    model_class = importlib.import_module(module_name).Model
    print("Model:", module_name, hyper_parameters)

    print("Training on ALL of the training data...")
    model = model_class(tqdm(data.ALL_TRAIN, dynamic_ncols=True), **hyper_parameters)
    print("Labelling unlabelled tweets...")
    for tweet in tqdm(data.TEST, dynamic_ncols=True):
        tweet.handle = model.predict(tweet)

    print("Saving submission...")
    current_time = time.strftime('%m-%d_%H-%M-%S')
    model_name = models.model_name(module_name, hyper_parameters)
    filename = f"../submissions/submission-{current_time}-{model_name}.csv"
    data.export(filename, data.TEST)
    print("Saved to:", filename)

if __name__ == '__main__':
    main()
