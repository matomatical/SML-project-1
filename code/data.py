import csv

class Tweet:
    def __init__(self, handle, text):
        self.handle = handle
        self.text = text
    def __repr__(self):
        return f"Tweet({self.handle!r}, {self.text!r})"
    def __str__(self):
        return f'@{self.handle} says: "{self.text}"'

# data contains no stray tabs or newlines:
# tabs are ONLY used to separate ids from tweets,
# newlines are ONLY used to terminate lines
# there's no other whitespace before/after any tweet
with open('../data/train_tweets.txt') as file:
    TRAIN = [Tweet(*line.strip().split('\t')) for line in file]

with open('../data/test_tweets_unlabeled.txt') as file:
    TEST = [Tweet('??????', line.strip()) for line in file]

def export(filename, tweets):
    with open(filename, 'w') as outfile:
        out = csv.writer(outfile)
        out.writerow(['Id', 'Predicted'])
        out.writerows(enumerate((tweet.handle for tweet in tweets), 1))
