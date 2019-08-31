import csv
from nltk import ngrams
import re

class Tweet:
    def __init__(self, handle, text):
        self.handle = handle
        self.text = text

        self.normalised_text = self.normalise_text(text)

        
    def __repr__(self):
        return f"Tweet({self.handle!r}, {self.text!r})"
    def __str__(self):
        return f'@{self.handle} says: "{self.text}"'

    def normalise_text(self, text):
        reg = [
            # Decision was made to keep "html tags", majority of uses are not actual html tags and are actually incredibly indicative of user
            # My favourite: v    v          v    v
            # ´·.¸¸.·´¯`·.¸><(((º>¸.·´¯`·.¸><(((º>
            # (r"<[^>]+>", "H"), 
            
            # Most of the patterns taken from: https://github.com/theocjr/social-media-forensics/blob/master/microblog_authorship_attribution/dataset_pre_processing/tagging_irrelevant_data.py
            # Order is important, because urls and etc can be converted to U, e.g. "@handle://www.powerwomenmagazine.com/" compare converting url before vs after hashtag conversion
            (r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '#'),                         # #hashtag
            (r'(?<!\S)@[0-9a-zA-Z_]{1,}(?![0-9a-zA-Z_])', '@'),              # @mention
            # URL 
            (r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)', 'U'),
            # Dates, although indistinguishable from listings of 2-3  numbers without full text analysis. e.g. 2-3 is considered a date, 10/20/30 is also 
            (r'(?<!\S)([0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]\s?[/-]\s?[0-9]{1,4}|[0-1]?[0-9]\s?[/-]\s?[0-9]{1,4}|[0-9]{1,4}\s?[/-]\s?[0-1]?[0-9]|[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9])(?![0-9a-zA-Z])', 'D'), 
            (r'[0-9]?[0-9]:[0-9]?[0-9](:[0-9]?[0-9])?', 'T'),                # time
            (r'(?:(?:\d+,?)+(?:\.?\d+)?)', 'N')                              # numbers
        ] # weird edge cases probably aren't covered, but honestly it's a lot of work for probably very few cases
        
        normalised_text = text
        for pattern, replacement in reg:
            normalised_text = re.sub(pattern, replacement, normalised_text)
        return normalised_text

    # includes option to use normalised text or not, default to true
    def char_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.text
        return [''.join(x) for x in ngrams(chosen_text, n, pad_left=True, left_pad_symbol=" ", pad_right=True, right_pad_symbol=" ")]

    def word_ngram(self, n, norm=True):
        chosen_text = self.normalised_text if norm else self.text
        # chosen_text = re.sub(r"[\W]+", " ", chosen_text) # Might be good to remove punctuations, brackets etc from the word gram, as they're captured in the char gram
        chosen_text = re.sub(r"[\[\\\^\.\|\?\(\),<>/;:'\"{}~`]", " ", chosen_text).split() # "purifies" the string so it's just words. probably. 
        return [x for x in ngrams(chosen_text, n, pad_left=True, left_pad_symbol="STT", pad_right=True, right_pad_symbol="END")]
        

# data contains no stray tabs or newlines:
# tabs are ONLY used to separate ids from tweets,
# newlines are ONLY used to terminate lines
# there's no other whitespace before/after any tweet
with open('../data/traditional_split/training_tweets.txt', encoding = 'utf8') as file:
    TRAIN = [Tweet(*line.strip().split('\t')) for line in file]

with open('../data/traditional_split/dev_tweets.txt', encoding = 'utf8') as file:
    DEVEL = [Tweet(*line.strip().split('\t')) for line in file]

with open('../data/test_tweets_unlabeled.txt', encoding = 'utf8') as file:
    TEST = [Tweet('??????', line.strip()) for line in file]

def export(filename, tweets):
    with open(filename, 'w') as outfile:
        out = csv.writer(outfile)
        out.writerow(['Id', 'Predicted'])
        out.writerows(enumerate((tweet.handle for tweet in tweets), 1))

