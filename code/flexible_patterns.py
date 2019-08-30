from collections import defaultdict
from data import Tweet, TRAIN 

HFW = 0
CW = 1

class FlexiblePattern:
  # pattern is a tuple of (word, label) tuples
  def __init__(self, pattern):
    self.pattern = pattern

  def __repr__(self):
    rep = []
    for word, label in self.pattern:
      if label == HFW:
        rep.append(f"('{word}', HFW)")
      else:
        rep.append(f"('{word}', CW)")
    str_rep = ", ".join(rep)
    return f"({str_rep})"
  
  # tuple of words where words labeled by CW are replaced by the label
  def tokenise_cw(self):
    tokenised = []
    for word, label in self.pattern:
      if label == HFW:
        tokenised.append(word)
      else:
        tokenised.append(CW)
    return tuple(tokenised)
  


def generate_hfw(tweets=TRAIN):
  word_freq = defaultdict(int)

  for tweet in tweets:
    for word in tweet.text.lower().split():
      word_freq[word] += 1
  hfw_threshold = len(word_freq) * 10e-4

  hfw = set() 

  for word in word_freq:
    if word_freq[word] > hfw_threshold:
      hfw.add(word)
  
  return hfw

high_freq_words = generate_hfw()

def flexible_patterns(tweet, lo=2, hi=6):
  modified_list = [] # list of (word, label) tuples
  for word in tweet.text.split():
    if word.lower() in high_freq_words:
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




  

