import sys
import time

from tqdm import tqdm

import data; data.load_test()
import models

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = models.parse_hyper_parameters(sys.argv[2:])
    model = models.load(module_name, hyper_parameters)
    print("Model:", module_name, hyper_parameters)

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
