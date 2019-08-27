import numpy as np
from collections import defaultdict
import time

def train(data):    

    ngramLen = 3
    L = 200

    model = SimpleNGramModel(ngramLen)
    
    for t in data:
        model.feed(t.handle, t.text)
    
    model.trim(L)

    return model

def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class SimpleNGramModel:
    def __init__(self, ngramLen):

        self.ngramLen = ngramLen

        self.ngrams = defaultdict(_ddictpickle) # {handle: {ngram: count, ...}, ...}
        # After triming converted to {handle: set(top_L_ngrams)}


    def feed(self, handle, text):
        for ngs in self.generateNgrams(text):
            self.ngrams[handle][ngs] += 1

    # keep L most frequent ngrams (including count?)
    def trim(self, L):
        for hg in self.ngrams.items():
            handle = hg[0]
            grams = hg[1]

            topL = sorted(grams.items(), key = lambda x: x[1], reverse = True)[:L]

            # If not keeping count: 
            # dict((n, c) for n,c in topL) -> set([n for n,_ in topL])
            self.ngrams[handle] = set([n for n,_ in topL])  

    def predict(self, tweet):
        
        tweetGrams = self.generateNgrams(tweet)
        tweetGramsSet = set(tweetGrams)
        highestMatch = 0
        highestHandle = "?????"

        for hn in self.ngrams.items():
            matchCount = len(tweetGramsSet.intersection(hn[1]))

            if matchCount > highestMatch:
                highestMatch = matchCount
                highestHandle = hn[0]

        return highestHandle

    def generateNgrams(self, text):
        n = self.ngramLen
        ngrams = []
        textPadded = ("_"*(n-1)) + text + ("_"*(n-1))
        for i in range(len(textPadded)-n+1):
            ngrams.append(textPadded[i:i+n])
                
        return ngrams
