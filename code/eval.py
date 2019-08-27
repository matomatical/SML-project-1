import sys

from tqdm import tqdm

import data
import models

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    model_name = sys.argv[1]
    model = models.load(model_name)
    print("Model:", model_name)

    correct = 0
    tests = 0
    for tweet in tqdm(data.DEVEL):
        predicted_handle = model.predict(tweet.text)
        if predicted_handle == tweet.handle:
            correct += 1
        tests += 1

    accuracy = correct / tests
    print(f"Label accuracy: {correct}/{tests} ({accuracy:%})")

if __name__ == '__main__':
    main()
