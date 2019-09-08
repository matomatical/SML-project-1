"""
data module: utilities for loading, representing,
pre-processing and processing our twitter data
(assumed to exist in ../data/ directory)
"""

import re
import sys
import csv
import pickle
from enum import Enum 
from collections import defaultdict

from nltk import ngrams


class Tweet:
    """Our internal representation of a tweet: a handle and text"""
    def __init__(self, handle, text):
        self.handle = handle
        self.raw_text = text
        self.normalised_text = normalise(tokenise(text))

    def __repr__(self):
        return f"Tweet({self.handle!r}, {self.raw_text!r})"
    def __str__(self):
        return f'@{self.handle} says: "{self.raw_text}"'

    def ngram(self, n, level, norm=True):
        if level == "char":
            return self.char_ngram(n, norm=norm)
        if level == "word":
            return self.word_ngram(n, norm=norm)
        if level == "byte":
            return self.byte_ngram(n, norm=norm)
        if level == "flex":
            return self.flexible_pattern(n, n, norm=norm)

    # includes option to use normalised text or not
    def char_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.raw_text
        return [''.join(x) for x in ngrams(chosen_text, n, pad_left=True, left_pad_symbol=" ", pad_right=True, right_pad_symbol=" ")]

    def word_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.raw_text
        chosen_split = chosen_text.split()
        return [x for x in ngrams(chosen_split, n, pad_left=True, left_pad_symbol="STT", pad_right=True, right_pad_symbol="END")]
        
    def byte_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.raw_text
        text_bytes = bytes(chosen_text, encoding="utf-8")
        return list(ngrams(text_bytes, n, pad_left=True, left_pad_symbol=b' ', pad_right=True, right_pad_symbol=b' '))

    USED_FLEX_PATTERNS = False
    def flexible_pattern(self, min_n, max_n, hfws=None, norm=True):
        if not Tweet.USED_FLEX_PATTERNS:
            if not norm:
                print("[data.py] WARNING: Flexible patterns do not support un-normalised mode. Using normalised flexible patterns instead")
            load_hfw()
            Tweet.USED_FLEX_PATTERNS = True
        return flexible_patterns(self, lo=min_n-1, hi=max_n)



"""
functions defining flexible patterns
YOU MUST FIRST RUN FLEXIBLE_PATTERNS.PY TO COUNT HIGH FREQUENCY WORDS
"""
 
class Label(Enum):
  HFW = 0;
  CW = 1
  def __repr__(self):
    return self.name
  def __str__(self):
    return self.name

HFW = Label.HFW
CW  = Label.CW

class FlexiblePattern:
  # pattern is a tuple of (word, label) tuples
  def __init__(self, pattern):
    # self.pattern = ("hfw1", CW, CW, "hfw2", CW, "hfw3") (for example)
    # tuple of words where words labeled by CW are replaced by the label
    self.pattern = []
    for word, label in pattern:
      if label == HFW:
        self.pattern.append(word)
      else:
        self.pattern.append(CW)
    self.pattern = tuple(self.pattern)

  def __repr__(self):
    return 'FlexiblePattern(' + " ".join(map(str, self.pattern)) + ')'

  def __eq__(self, other):
    return self.pattern == other.pattern
  def __hash__(self):
    return hash(self.pattern)
  
  def tokenise_cw(self):
    tokenised = []
    for word, label in self.pattern:
      if label == HFW:
        tokenised.append(word)
      else:
        tokenised.append(CW)
    return tuple(tokenised)


HFW_SET = None

def load_hfw():
    global HFW_SET
    if HFW_SET is None:
        try:
            with open('../data/hfws.pickle', 'rb') as file:
                HFW_SET = pickle.load(file)
        except Exception as e:
            print("Error!", e) 
            print("(did you run flexible_patterns.py to generate this data file?)")


def flexible_patterns(tweet, lo=2, hi=6):
  modified_list = [] # list of (word, label) tuples
  for word in tweet.normalised_text.lower().split():
    if word in HFW_SET:
      modified_list.append((word, HFW))
    else:
      modified_list.append((word, CW))

  flexible_patterns = []
  for i in range(len(modified_list)):
    _, label = modified_list[i]

    if label == HFW:
      HFW_count = 1
      j = i+1

      # creates a flexible pattern of the next "n" HFW, where lo < n < hi
      while j < len(modified_list) and HFW_count < hi:
        _, label = modified_list[j]
        if label == HFW:
          HFW_count += 1
          if HFW_count > lo:
            flexible_patterns.append(FlexiblePattern(tuple(modified_list[i:j+1])))
        j += 1
  
  return flexible_patterns




"""
functions for pre-processing tweets (split words/punctutation,
normalise sparse features like URLs, and more)
"""
def tokenise(tweet_text):
    #                      v html escape code '&#?\w+;'
    #                               v punctuation symbols [.,!?:;"'~(){}[]<>*=-]
    PUNCT_GROUPS    = r"(((&#?\w+;)|[\.\,\!\?\:\;\"\'\~\(\)\{\}\[\]\<\>\*\=\-])+)"

    # first separate punctuation groups from the end of each word
    PUNCT_AT_END    = re.compile(PUNCT_GROUPS + r"\s"), r" \g<1> "
    # then take it away from the start too :)
    PUNCT_AT_START  = re.compile(r"\s" + PUNCT_GROUPS), r" \g<1> "
    # if there is only one piece of punct in the middle (so "that's"
    # or "wishy-washy" but not "https://example.com") then split that too!
    PUNCT_IN_MIDDLE = re.compile(r"\s(\w+)" + PUNCT_GROUPS + r"(\w+)(?=\s)"), r" \g<1> \g<2> \g<5>"
    # maybe we introduced some double spaces with the above; remove them.
    EXTRA_SPACES    = re.compile(r"\s\s*"), r" "
    # pad with spaces to simplify expressions (spaces are word boundaries)
    text = f" {tweet_text} "
    # apply a sequence of pattern substitutions
    sequence = [PUNCT_AT_END, PUNCT_AT_START, PUNCT_IN_MIDDLE, EXTRA_SPACES]
    for pattern, replacement in sequence:
        text = pattern.sub(replacement, text)
    return text

def normalise(tweet_text):
    # Patterns for normalising, and their corresponding replacement
    # Decision was made to keep "html tags", majority of uses are not
    # actual html tags and are actually incredibly indicative of user
    # Patterns obtained from: 
    #       https://github.com/theocjr/social-media-forensics/blob/master/microblog_authorship_attribution/dataset_pre_processing/tagging_irrelevant_data.py
    HASHTAG = re.compile(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)'), '#'

    MENTION = re.compile(r'(?<!\S)@[0-9a-zA-Z_]{1,}(?![0-9a-zA-Z_])'), '@'

    URL     = re.compile(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?'
                         r'[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)'
                         r'[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??'
                         r'(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)'), 'U'

    DATE    = re.compile(r'(?<!\S)([0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]\s?[/-]\s?'
                         r'[0-9]{1,4}|[0-1]?[0-9]\s?[/-]\s?[0-9]{1,4}|[0-9]{1,4}'
                         r'\s?[/-]\s?[0-1]?[0-9]|[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9])'
                         r'(?![0-9a-zA-Z])'), 'D'

    TIME    = re.compile(r'[0-9]?[0-9]:[0-9]?[0-9](:[0-9]?[0-9])?'), 'T'

    NUMS    = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)'), 'N'
    
    sparse_patterns = [HASHTAG, MENTION, URL, DATE, TIME, NUMS]
    
    normalised_text = tweet_text
    for pattern, replacement in sparse_patterns:
        normalised_text = re.sub(pattern, replacement, normalised_text)
    return normalised_text



"""
Load the data, lazily
"""
TRAIN = None
ALL_TRAIN = None
DEVEL = None
TEST  = None
DEVEL1000 = None

def load_train():
    global TRAIN
    if TRAIN is None:
        print("[data.py] Loading ../data/traditional_split/training_tweets.txt into TRAIN")
        with open('../data/traditional_split/training_tweets.txt') as file:
            TRAIN = [Tweet(*line.strip().split('\t')) for line in file]

def load_all_train():
    global ALL_TRAIN
    if ALL_TRAIN is None:
        print("[data.py] Loading ../data/train_tweets.txt into ALL_TRAIN")
        with open('../data/train_tweets.txt') as file:
            ALL_TRAIN = [Tweet(*line.strip().split('\t')) for line in file]
        

def load_devel():
    global DEVEL
    if DEVEL is None:
        print("[data.py] Loading ../data/traditional_split/dev_tweets.txt into DEVEL")
        with open('../data/traditional_split/dev_tweets.txt') as file:
            DEVEL = [Tweet(*line.strip().split('\t')) for line in file]

def load_devel1000():
    global DEVEL1000
    if DEVEL1000 is None:
        print("[data.py] Loading ../data/traditional_split/1000_dev_tweets.txt into DEVEL1000")
        with open('../data/traditional_split/1000_dev_tweets.txt') as file:
            DEVEL1000 = [Tweet(*line.strip().split('\t')) for line in file]

def load_test():
    global TEST
    if TEST is None:
        print("[data.py] Loading ../data/test_tweets_unlabeled.txt into TEST")
        with open('../data/test_tweets_unlabeled.txt') as file:
            TEST = [Tweet('??????', line.strip()) for line in file]

# Notes: 
# data contains no stray tabs or newlines:
# tabs are ONLY used to separate ids from tweets,
# newlines are ONLY used to terminate lines
# there's no other whitespace before/after any tweet

def export(filename, tweets):
    with open(filename, 'w') as outfile:
        out = csv.writer(outfile)
        out.writerow(['Id', 'Predicted'])
        out.writerows(enumerate((tweet.handle for tweet in tweets), 1))

