import numpy as np
from collections import defaultdict
import heapq


def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)


class Model:
    def __init__(self, data, n, L, norm, level="char"):

        self.ngramLen = int(n)
        self.L = int(L)
        self.norm = eval(norm)
        self.level = level

        self.ngrams = defaultdict(_ddictpickle) # {handle: {ngram: count, ...}, ...}
        # After triming converted to {handle: set(top_L_ngrams)}

        self.invertedNgram = defaultdict(set) # {ngram: set(handles), ...} used for inverted index

        for t in data:
            for ng in t.ngram(self.ngramLen, norm=self.norm, level=self.level):
                self.ngrams[t.handle][ng] += 1
        
        self.trim(self.L)

    # keep L most frequent ngrams (including count?)
    def trim(self, L):
        for hg in self.ngrams.items():
            handle = hg[0]
            grams = hg[1]

            topL = sorted(grams.items(), key = lambda x: x[1], reverse = True)[:L]

            # If not keeping count: 
            # dict((n, c) for n,c in topL) -> set([n for n,_ in topL])
            self.ngrams[handle] = set([n for n,c in topL if c > 1])  

        # Inverting index
        for hg in self.ngrams.items():
            handle = hg[0]
            for g in hg[1]:
                self.invertedNgram[g].add(handle)

    def predict(self, tweet, topk=None, scores=False):
        tweetGrams = tweet.ngram(self.ngramLen, norm=self.norm, level=self.level)
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
            if topk is None:
                return "?????" # unknown
            else:
                return []
        
        # add more flexible output modes for ensemble methods
        # (default behaviour remains the same)
        if topk is None:
            best_pair = max(matches.items(), key=lambda x: x[1])
            if scores:
                return best_pair
            else:
                return best_pair[0]
        else:
            best_pairs = heapq.nlargest(topk, matches.items(), key=lambda x: x[1])
            if scores:
                return best_pairs
            else:
                return [handle for handle, score in best_pairs]
        # return max(matches.items(), key = lambda x: x[1])[0]
        # return sorted(matches.items(), key = lambda x : x[1], reverse = True)[0]
