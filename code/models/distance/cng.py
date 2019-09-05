from collections import defaultdict

def _ddictpickle(): # needed to pickle the module
    return defaultdict(int)

class Model:

    def __init__(self, data, n, L, norm, level='char'):
        
        self.ngramLen = int(n)
        self.L = int(L)
        self.norm = bool(eval(norm))
        self.level = level

        self.ngrams = defaultdict(_ddictpickle) # {handle: {ngram: count, ...}}
        self.totals = defaultdict(int)
        # self.sets = {}

        for t in data:

            for ng in t.ngram(self.ngramLen, norm=self.norm, level=self.level):
                self.ngrams[t.handle][ng] += 1
                self.totals[t.handle] += 1
        
            # print(self.ngrams)
            # break

        self.trim(self.L)

        # for handle, ngrams in self.ngrams.items():
        #     self.sets[handle] = set([n for n,_ in ngrams.items()])

    def trim(self, L):
        for handle, grams in self.ngrams.items():
            
            topL = {}
            
            for ngram, count in sorted(grams.items(), key = lambda x: x[1], reverse = True)[:L]:
                topL[ngram] = count/self.totals[handle]
            if int(handle) == 3411:
                print(topL)
            self.ngrams[handle] = topL # convert it to static dict
        
    # def distance(self, p1, sp1, p2, sp2):
    def distance(self, p1, p2):
        k = 0
        # problem is we have to do the union of two sets of ngrams, can't just iterate through p1 then p2
        # Xp1p2 = sp1.union(sp2) 

        # for x in Xp1p2:
        #     p1x = 0 if x not in p1 else p1[x]
        #     p2x = 0 if x not in p2 else p2[x]
        #     k += ( (2*(p1x-p2x)) / (p1x+p2x) ) ** 2

        # doing this is faster than doing set union
        # doesn't need sp1, sp2 in inputs
        encountered = set() 

        for x,c in p1.items():
            if x not in encountered:
                encountered.add(x)
                p1x = c
                p2x = 0 if x not in p2 else p2[x]
                k += ( (2*(p1x-p2x)) / (p1x+p2x) ) ** 2

        for x,c in p2.items():
            if x not in encountered:
                encountered.add(x)
                p1x = 0 if x not in p1 else p1[x]
                p2x = c
                k += ( (2*(p1x-p2x)) / (p1x+p2x) ) ** 2

        return k

    def predict(self, tweet):

        p2 = {} # Accessing by an index in default dict creates an entry with value 0, so don't use it
        p2_total = 0
        # sp2 = set()
        for ngram in tweet.ngram(self.ngramLen, norm=self.norm, level=self.level):
            p2[ngram] = 1 if ngram not in p2 else p2[ngram] + 1
            # sp2.add(ngram)
            p2_total += 1

        for ngram, count in p2.items():
            p2[ngram] = count/p2_total

        h_min = '0'
        k_min = 999999999

        # for h, sp1 in self.sets.items():
        for h, p1 in self.ngrams.items():
            # print(h)
            # k_curr = self.distance(self.ngrams[h], sp1, p2, sp2)
            k_curr = self.distance(p1, p2)
            if k_curr < k_min:
                h_min = h
                k_min = k_curr
                # print(k_curr)

        print(h_min)
        # Gravitates to predicting the absolute shortest authors
        # e.g. 3411, 3983
        return h_min
        