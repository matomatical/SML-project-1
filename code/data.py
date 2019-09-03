import csv
from nltk import ngrams
import re

class Tweet:
    def __init__(self, handle, text):
        self.handle = handle
        self.raw_text = text
        self.normalised_text = normalise(tokenise(text))

    def __repr__(self):
        return f"Tweet({self.handle!r}, {self.raw_text!r})"
    def __str__(self):
        return f'@{self.handle} says: "{self.raw_text}"'

    # includes option to use normalised text or not
    def char_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.raw_text
        return [''.join(x) for x in ngrams(chosen_text, n, pad_left=True, left_pad_symbol=" ", pad_right=True, right_pad_symbol=" ")]

    def word_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.raw_text
        # chosen_text = re.sub(r"[\W]+", " ", chosen_text) # Might be good to remove punctuations, brackets etc from the word gram, as they're captured in the char gram
        # chosen_text = re.sub(r"[\[\\\^\.\|\?\(\),<>/;:'\"{}~`]", " ", chosen_text).split() # "purifies" the string so it's just words. probably. 
        chosen_split = chosen_text.split()
        return [x for x in ngrams(chosen_split, n, pad_left=True, left_pad_symbol="STT", pad_right=True, right_pad_symbol="END")]
        
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
DEVEL = None
TEST  = None

def load_train():
    global TRAIN
    if TRAIN is None:
        print(" Loading ../data/traditional_split/training_tweets.txt into TRAIN")
        with open('../data/traditional_split/training_tweets.txt') as file:
            TRAIN = [Tweet(*line.strip().split('\t')) for line in file]

def load_devel():
    global DEVEL
    if DEVEL is None:
        print(" Loading ../data/traditional_split/dev_tweets.txt into DEVEL")
        with open('../data/traditional_split/dev_tweets.txt') as file:
            DEVEL = [Tweet(*line.strip().split('\t')) for line in file]

def load_test():
    global TEST
    if TEST is None:
        print(" Loading ../data/test_tweets_unlabeled.txt into TEST")
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
