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

    for tweet in tqdm(data.TEST):
        tweet.handle = model.predict(tweet.text)

    data.export("peace_love_yoga.csv", data.TEST)

if __name__ == '__main__':
    main()
