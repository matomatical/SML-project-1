from collections import defaultdict

import time

def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class Model:

    def __init__(self, data, n, L, norm, level='char'):
        
        self.ngramLen = int(n)
        self.L = int(L)
        self.norm = bool(eval(norm))
        self.level = level

        self.ngrams = defaultdict(_ddictpickle) # {handle: {ngram: count, ...}}
        self.invertedNgrams = defaultdict(_ddictpickle) # {ngram: {handle: count, ...}}
        self.sets = {}
        self.handles = set()

        self.num_tweets = defaultdict(int)

        for t in data:
            self.handles.add(t.handle)
            for ng in t.ngram(self.ngramLen, norm=self.norm, level=self.level):
                self.ngrams[t.handle][ng] += 1

            self.num_tweets[t.handle] += 1

        self.trim(self.L)

        # add ngrams by author to set for faster access, for set intersection
        for handle, ngrams in self.ngrams.items():
            self.sets[handle] = set([n for n,_ in ngrams.items()])

        # invert index
        for handle, ngrams in self.ngrams.items():
            for ng,_ in ngrams.items():
                self.invertedNgrams[ng][handle] += 1

        # convert to static dict so it doesn't add values when accessing non-key
        for ngrams, handles in self.invertedNgrams.items():
            self.invertedNgrams[ngrams] = dict(handles)

        self.invertedNgrams = dict(self.invertedNgrams) 


    def trim(self, L):
        for handle, grams in self.ngrams.items():
            
            topL = {}
            topL_sum = sum([c for _,c in grams.items()])
            
            for ngram, count in sorted(grams.items(), key = lambda x: x[1], reverse = True)[:L]:
                topL[ngram] = count/topL_sum
            # if int(handle) == 3411:
            #     print(topL)
            self.ngrams[handle] = topL # convert it to static dict
        
    def distances(self, p2, sp2):
    # def distance(self, p1, p2):
        # Calculate distance from tweet ngrams to authors that used those ngrams

        # Instead of union of two sets of ngrams:
        # Formula works out to k = 4 if one set doesn't have an ngram
        # Formula also seemingly assumes all authors have at least L n-grams, based on predictive behaviour
        # Sometimes we don't have L ngrams
        # To make it more flexible, we calculate according to:
        #   (L+T-2N) * 4 + dist(x)_(for all x in N)
        #   Where L is forced profile len, T is tweet, N is intersection, x is ngram
        #   L is not forced on tweet ngram length because that doesn't make sense for the range of L's used in tuning

        L = self.L
        T = len(sp2)
        # N = sp1.intersection(sp2) # Have to calculate for all handles
        # Ns = {}
        seen = set()
        Ks = {} # {handle: distance, ...}
        for ng in sp2:
            # print(ng)
            if ng in self.invertedNgrams:
                # print("PPPPe")
                for handle,_ in self.invertedNgrams[ng].items():
                    if handle not in seen:
                        seen.add(handle)
                        N = self.sets[handle].intersection(sp2) # intersection for every handle that uses the n-grams in Tweet p2

                        k = 0
                    
                        for x in N:
                            k += ((2 * self.ngrams[handle][x] - p2[x]) / (self.ngrams[handle][x] + p2[x])) ** 2

                        Ks[handle] = (L + T - 2 * len(N)) + k

        return Ks


    def distance2(self, p1, sp1, p2, sp2):
        L = self.L
        T = len(sp2)
        N = sp1.intersection(sp2)

        k = (L + T - 2 * len(N)) * 4

        for x in N:
            k += ((2 * p1[x] - p2[x]) / (p1[x] + p2[x]) ** 2)

        return k


    def predict(self, tweet):
        final_prediction = 0
        p2 = {} # Accessing by an index in default dict creates an entry with value 0, so don't use it
        p2_total = 0
        sp2 = set()
        for ngram in tweet.ngram(self.ngramLen, norm=self.norm, level=self.level):
            p2[ngram] = 1 if ngram not in p2 else p2[ngram] + 1
            sp2.add(ngram)
            p2_total += 1

        for ngram, count in p2.items():
            p2[ngram] = count/p2_total

        distances = self.distances(p2, sp2)

        _, dist = min(distances.items(), key = lambda x: x[1])

        # of the authors with distance `dist`, predict the author with the most tweets
        predicted_author = (None, 0) # (handle, tweet count)
        for h in distances:
            if distances[h] == dist:
                if self.num_tweets[h] > predicted_author[1]:
                    predicted_author = (h, self.num_tweets[h])
        return predicted_author[0]

        # h_min = '0'
        # k_min = 999999999

        # # for h, sp1 in self.sets.items():
        # for h, p1 in self.ngrams.items():
        #     # print(h)
        #     # k_curr = self.distance(self.ngrams[h], sp1, p2, sp2)
        #     sp1 = self.sets[h]
        #     k_curr = self.distance2(p1, sp1, p2, sp2)
        #     if k_curr < k_min:
        #         h_min = h
        #         k_min = k_curr
        #         # print(k_curr)
        # return h_min
        