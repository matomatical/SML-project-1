import sys
import importlib

from tqdm import tqdm

import numpy as np

import data
import eval
import models

SEED = 51

def main():
    if len(sys.argv) <= 2:
        print("please specify a number of examples and a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)

    eval_set_size = int(sys.argv[1])
    module_name = sys.argv[2]

    # splitting training data
    print("splitting training data into", eval_set_size, "(test) v. rest (train)")
    data.load_train()
    tweets = np.array(data.TRAIN)
    np.random.seed(SEED)
    np.random.shuffle(tweets)
    test_tweets, train_tweets = tweets[:eval_set_size], tweets[eval_set_size:]
    
    hyper_parameters = models.parse_hyper_parameters(sys.argv[3:])
    model_class = importlib.import_module(module_name).Model
    print("Model:", module_name, hyper_parameters)

    print("Training...")
    model = model_class(tqdm(train_tweets, dynamic_ncols=True), **hyper_parameters)
    print("Evaluating...")
    accuracy, correct, tests = eval.evaluate(model, tqdm(test_tweets, dynamic_ncols=True))
    print(f"Label accuracy: {correct}/{tests} ({accuracy:%})")


if __name__ == '__main__':
    main()
