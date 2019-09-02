import sys

from tqdm import tqdm

import data; data.load_devel()
import models

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = models.parse_hyper_parameters(sys.argv[2:])
    model = models.load(module_name, hyper_parameters)
    print("Model:", module_name, hyper_parameters)

    accuracy, correct, tests = evaluate(model, data.DEVEL)

    print(f"Label accuracy: {correct}/{tests} ({accuracy:%})")

def evaluate(model, data):
    correct = 0
    tests = 0
    for tweet in tqdm(data):
        predicted_handle = model.predict(tweet)
        if predicted_handle == tweet.handle:
            correct += 1
        tests += 1

    accuracy = correct / tests
    return accuracy, correct, tests


if __name__ == '__main__':
    main()
