import pickle
from collections import defaultdict

import data
data.load_all_train()
data.load_test()

FLEXIBLE_PATTERN_RATIO = 1e-4

def main():
  print("counting all words from the whole dataset...")
  word_freq = defaultdict(int)
  total_num_words = 0

  for tweet in data.ALL_TRAIN + data.TEST:
    for word in tweet.normalised_text.lower().split():
      word_freq[word] += 1
      total_num_words += 1
  print("counted!")
  
  print("defining high frequency words...")
  hfw_threshold = total_num_words * FLEXIBLE_PATTERN_RATIO
  hfw = set()

  for word in word_freq:
    if word_freq[word] > hfw_threshold:
      hfw.add(word)
  
  # save to file
  print("saving to ../data/hfws.pickle...")
  with open("../data/hfws.pickle", 'wb') as file:
    pickle.dump(hfw, file)
  print("done!")

if __name__ == '__main__':
    main()
