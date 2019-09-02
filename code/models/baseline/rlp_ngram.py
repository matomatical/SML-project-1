import numpy as np
from collections import defaultdict, Counter


def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class Model:
    def __init__(self, data, n, L):

        self.n = int(n)
        self.L = int(L)

        self.ngrams = defaultdict(_ddictpickle) # {handle: {ngram: count, ...}, ...}
        # After triming converted to {handle: set(top_L_ngrams)}

        self.invertedNgram = defaultdict(set) # {ngram: set(handles), ...} used for inverted index

        self.meanFrequencies = defaultdict(float)
        for t in data:
            self.feed(t)
        
        total_grams = sum(self.meanFrequencies.values())
        for gram in self.meanFrequencies:
            self.meanFrequencies[gram] = self.meanFrequencies[gram] / total_grams
        
    def feed(self, tweet):
        for ngs in tweet.char_ngram(self.n):
            self.ngrams[tweet.handle][ngs] += 1
            self.meanFrequencies[ngs] += 1

    # keep L most frequent ngrams (including count?)
    def trim(self, L):
        recentered = {}

        for handle, grams in self.ngrams.items():
            recentered = {}
            total_grams = sum(grams.values())
            for gram in grams:
                recentered[gram] = abs(grams[gram]/total_grams - self.meanFrequencies.get(gram, 0.0))

            topL = sorted(recentered.items(), key = lambda x: x[1], reverse = True)[:L]

            self.ngrams[handle] = {(n,grams[n]/total_grams) for n,_ in topL}

        # Inverting index
        for handle, grams in self.ngrams.items():
            for n, n_f in grams:
                self.invertedNgram[n].add((handle, n_f))

    def predict(self, tweet):
        tweetGrams = tweet.char_ngram(self.n)
        tweetGramsCounter = Counter(tweetGrams)
        


        # {handle: count}
        distances = defaultdict(int)

        for gram in tweetGramsCounter:
            if gram not in self.invertedNgram: # unknown ngram, skip
                continue
            
            author_n_f = tweetGramsCounter[gram]/sum(tweetGramsCounter.values())
            corpus_n_f = self.meanFrequencies[gram]
            # retrieve all handles with this ngram
            for h, n_f in self.invertedNgram[gram]:
                if (author_n_f < corpus_n_f == n_f < corpus_n_f):
                    distances[h] += 1
                else:
                    distances[h] -= 1        
        if len(distances) == 0:
            return "?????" # unknown 



        return max(distances.items(), key = lambda x: x[1])[0]

