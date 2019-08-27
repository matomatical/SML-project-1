import numpy as np
from collections import defaultdict
import time

def train(data):    

    ngramLen = 3
    L = 2000

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

        self.invertedNgram = defaultdict(set) # {ngram: set(handles), ...} used for inverted index


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

        # Inverting index
        for hg in self.ngrams.items():
            handle = hg[0]
            for g in hg[1]:
                self.invertedNgram[g].add(handle)

    def predict(self, tweet):
        # start = time.time()
        tweetGrams = self.generateNgrams(tweet)
        tweetGramsSet = set(tweetGrams)

        # {handle: count}
        matches = defaultdict(int)

        for gram in tweetGramsSet:
            if gram not in self.invertedNgram: # unknown ngram, skip
                continue
            
            # retrieve all handles with this ngram
            for h in self.invertedNgram[gram]:
                matches[h] += 1
        
        if len(matches) == 0:
            return "?????" # unknown 
        
        # end = time.time()
        # print(end-start)
        # print(match)
        return max(matches.items(), key = lambda x: x[1])[0]
        # return sorted(matches.items(), key = lambda x : x[1], reverse = True)[0]

    def generateNgrams(self, text):
        n = self.ngramLen
        ngrams = []
        textPadded = ("_"*(n-1)) + text + ("_"*(n-1))
        for i in range(len(textPadded)-n+1):
            ngrams.append(textPadded[i:i+n])
                
        return ngrams
