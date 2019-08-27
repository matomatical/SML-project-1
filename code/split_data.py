from data import TRAIN 
from itertools import groupby

import sklearn.model_selection as model_selection
import random

def generate_split_data(data=TRAIN, shuffle=True):
  train = []
  dev = []

  # group tweets by handle. Place 80% of group in train, and 20% in dev. 
  for _, g in groupby(data, lambda x : x.handle):
    samples = list(g)
    if (len(samples) == 1):
      train.extend(samples)
      continue

    train_sample, dev_sample = model_selection.train_test_split(samples, train_size=0.8, random_state=42)
    train.extend(train_sample)
    dev.extend(dev_sample)
  if (shuffle):
    random.Random(4).shuffle(dev)
    random.Random(5).shuffle(train)
  
  return train, dev

def export_split_data(train_filename, dev_filename, tweets):
  train, dev = generate_split_data();
  with open(train_filename, 'w') as outfile:
    outfile.writelines([tweet.handle + "\t" + tweet.text + "\n"  for tweet in train])

  with open(dev_filename, 'w') as outfile:
    outfile.writelines([tweet.handle + "\t" + tweet.text + "\n" for tweet in dev])

