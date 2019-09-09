import ast
import numpy as np
from collections import defaultdict, Counter
from heapq import nlargest
from scipy.spatial import distance


def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class Model:
    def __init__(self, data, n, L, level, norm):

        self.n = int(n)
        self.L = int(L)
        self.level = level
        self.norm = ast.literal_eval(norm)

        # {handle: {ngram: count, ...}, ...}
        # After triming converted to {handle: {ngram: recentered}}
        self.ngrams = defaultdict(_ddictpickle) 
        
        # {ngram: set(handles), ...} used for inverted index
        self.invertedNgram = defaultdict(set) 

        # {handle: num_tweets, ...} used for tie breakers
        self.num_tweets = defaultdict(int)

        # {ngram: mean frequency, ...}
        self.meanFrequencies = defaultdict(float)

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

        for handle, grams in self.ngrams.items():
            recentered = {}
            total_grams = sum(grams.values())
            for gram in grams:
                recentered[gram] = abs(grams[gram]/total_grams - self.meanFrequencies.get(gram, 0.0))

            # topL = sorted(recentered.items(), key = lambda x: x[1], reverse = True)[:L]

            self.ngrams[handle] = recentered

        # Inverting index
        for handle, grams in self.ngrams.items():
            for n, n_f in grams.items():
                self.invertedNgram[n].add((handle, n_f))

    def predict(self, tweet):
        tweetGrams = tweet.ngram(self.n, self.level, norm = self.norm)
        tweetGramsCounter = Counter(tweetGrams)
        
        distances = defaultdict(float)

        # transform counts to recentered normalised frequencies
        tweet_rnf = {}
        total = sum(tweetGramsCounter.values())
        for gram in tweetGramsCounter:
            tweet_rnf[gram] = tweetGramsCounter[gram]/total

        for author in self.ngrams:
            # all n-grams of the top L of either profile
            a_top_L = {ng for ng, _ in nlargest(self.L, self.ngrams[author].items(), key = lambda x: x[1])}
            t_top_L = {ng for ng, _ in nlargest(self.L, tweet_rnf.items(), key = lambda x: x[1])}

            top_L = a_top_L | t_top_L

            P_a = []
            P_t = []
            for ng in top_L:
                P_a.append(self.ngrams[author].get(ng, 0))
                P_t.append(tweet_rnf.get(ng, 0.))
            P_a = np.array(P_a)
            P_t = np.array(P_t)

            distances[author] = 1 - distance.cosine(P_t, P_a)

        _, dist = min(distances.items(), key = lambda x: x[1])

        # of the authors with distance `dist`, predict the author with the most tweets
        predicted_author = (None, None) # (handle, tweet count)
        for h in distances:
            if distances[h] == dist:
                if predicted_author[1] is None or self.num_tweets[h] > predicted_author[1]:
                    predicted_author = (h, self.num_tweets[h])
        return predicted_author[0]

