import ast
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from heapq import nlargest
from copy import deepcopy


def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class Model:
    def __init__(self, data, n, L, level, norm):

        self.n = int(n)
        self.L = int(L)
        self.level = level
        self.norm = ast.literal_eval(norm)

        # {handle: {ngram: count, ...}, ...}
        # After triming converted to {handle: set(top_L_ngrams)}
        self.ngrams = defaultdict(_ddictpickle) 
        
        # {ngram: set(handles), ...} used for inverted index
        self.invertedNgram = defaultdict(set) 

        # {handle: num_tweets, ...} used for tie breakers
        self.num_tweets = defaultdict(int)

        # {ngram: mean frequency, ...}
        self.meanFrequencies = defaultdict(float)

        # {handle: sum [P(x) - E(x)], ...}
        self.offset = defaultdict(float)

        for t in data:
            self.num_tweets[t.handle] += 1
            for ng in t.ngram(self.n, self.level, norm=self.norm):
                self.ngrams[t.handle][ng] += 1
                self.meanFrequencies[ng] += 1

        total_grams = sum(self.meanFrequencies.values())
        for gram in self.meanFrequencies:
            self.meanFrequencies[gram] = self.meanFrequencies[gram] / total_grams
        
        self.trim(self.L)
            
    # keep L most distinctive ngrams (including count?)
    def trim(self, L):
        recentered = {}

        corpus_top_L = dict(nlargest(L, self.meanFrequencies.items(), key = lambda x: x[1]))
        for handle, grams in tqdm(self.ngrams.items()):
            recentered = deepcopy(corpus_top_L)
            total_grams = sum(grams.values())

            for gram in grams:
                recentered[gram] = abs(grams.get(gram, 0)/total_grams - self.meanFrequencies[gram])                

            topL = sorted(recentered.items(), key = lambda x: x[1], reverse = True)[:L]

            total_grams = sum(grams[gram] for gram, _ in topL)
            for gram, _ in topL:
                if grams[gram]/total_grams > self.meanFrequencies.get(gram, 0.0):
                    self.offset[handle] -= 1
                else:
                    self.offset[handle] += 1

            self.ngrams[handle] = {(n,grams[n]/total_grams) for n,_ in topL}

        # Inverting index
        for handle, grams in self.ngrams.items():
            for n, n_f in grams:
                self.invertedNgram[n].add((handle, n_f))

    def predict(self, tweet):
        tweetGrams = tweet.ngram(self.n, self.level, norm = self.norm)
        tweetGramsCounter = Counter(tweetGrams)

        distances = defaultdict(int)

        for gram in tweetGramsCounter:
            if gram not in self.invertedNgram: # unknown ngram, skip
                continue
            
            author_n_f = tweetGramsCounter[gram]/sum(tweetGramsCounter.values())
            corpus_n_f = self.meanFrequencies[gram]
            # retrieve all handles with this ngram
            for h, n_f in self.invertedNgram[gram]:
                if (author_n_f > corpus_n_f and n_f > corpus_n_f):
                    distances[h] += 3
                else:
                    distances[h] -= 1  
        for handle in self.offset:
            distances[handle] +=  self.offset[handle]     
        if len(distances) == 0:
            return "?????" # unknown 

        _, dist = min(distances.items(), key = lambda x: x[1])

        # of the authors with distance `dist`, predict the author with the most tweets
        predicted_author = (None, 0) # (handle, tweet count)
        for h in distances:
            if distances[h] == dist:
                if self.num_tweets[h] > predicted_author[1]:
                    predicted_author = (h, self.num_tweets[h])

        print(tweet)
        print(predicted_author[0])
        return predicted_author[0]

