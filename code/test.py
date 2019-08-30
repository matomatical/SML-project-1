import sys
import time

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

    print("Labelling unlabelled tweets...")
    for tweet in tqdm(data.TEST):
        tweet.handle = model.predict(tweet.text)

    print("Saving submission...")
    filename = f"../submissions/submission-{time.strftime('%m-%d_%H-%M-%S')}.csv"
    data.export(filename, data.TEST)
    print("Saved to:", filename)

if __name__ == '__main__':
    main()
