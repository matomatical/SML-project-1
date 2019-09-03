import math
from collections import defaultdict, Counter

class Model:
    def __init__(self, data, n, level="char", k=0, bag="True", prior="True", norm="True"):
        self.n = int(n)
        self.level = level
        self.k = float(k)
        self.useNormalised = eval(norm)
        self.useBagOfWords = eval(bag)
        self.usePriorDist  = eval(prior)

        # collect raw counts
        # self.author_counts :: {author: {ngram: count (from particular author's tweets)}}
        self.author_counts = defaultdict(Counter)
        # self.corpus_counts :: {ngram: count (over ALL tweets)}
        self.corpus_counts = Counter()
        # self.author_ntweets :: {author: number of tweets}
        self.author_ntweets = defaultdict(int)
        for tweet in data:
            ngrams = tweet.char_ngram(n=self.n, norm=self.useNormalised)
            self.author_counts[tweet.handle].update(ngrams)
            self.corpus_counts.update(ngrams)
            self.author_ntweets[tweet.handle] += 1
        # self.corpus_ntweets :: number of tweets in whole corpus
        self.corpus_ntweets = sum(self.author_ntweets.values())
        self.authors = set(self.author_ntweets)
        self.nauthors = len(self.authors)

        # normalise counts per profile and prior probabilities
        # self.author_prior :: {author: prob of author tweeting}
        self.author_prior = {}
        # self.author_probs :: {author: {ngram: prob of ngram for this author}}
        self.author_probs = {}
        for author in self.authors:
            self.author_prior[author] = self.author_ntweets[author] / self.corpus_ntweets
            profile = self.author_counts[author]
            nngrams = sum(profile.values())
            self.author_probs[author] = defaultdict(float)
            self.author_probs[author].update((ngram, count/nngrams) for ngram, count in profile.items())

        # construct inverted index for quicker querying later
        # TODO

    def predict(self, tweet):
        # featurise the tweet
        ngrams = tweet.char_ngram(n=self.n, norm=self.useNormalised)
        counts = Counter(ngrams)
        
        # find the nearest author according to log-probability distance metric
        best_author, best_distance = "??????", math.inf
        for author in self.authors:
            distance = 0
            for ngram, count in counts.items():
                prob_author_ngram = self.author_probs[author][ngram]
                if prob_author_ngram == 0:
                    distance = math.inf
                    break
                distance += count * math.log(prob_author_ngram)
            prior_prob = self.author_prior[author]
            distance += math.log(prior_prob)
            # actually, this is the negative distance, we need to negate
            distance = -distance
            if distance < best_distance:
                best_author, best_distance = author, distance
        
        return best_author
                
