import math
from enum import Enum
from collections import defaultdict, Counter

class Model:
    def __init__(self, data, n, level="char", smoothing="none", prior="True", norm="True"):
        self.n = int(n)
        self.level = level
        self._interpret_smoothing(smoothing)
        self.useNormalised = eval(norm)
        self.usePriorDist  = eval(prior)

        # collect raw counts
        # self.author_counts :: {author: {ngram: count (from particular author's tweets)}}
        self.author_counts = defaultdict(Counter)
        # self.corpus_counts :: {ngram: count (over ALL tweets)}
        self.corpus_counts = Counter()
        # self.author_ntweets :: {author: number of tweets}
        self.author_ntweets = defaultdict(int)
        for tweet in data:
            ngrams = tweet.ngram(n=self.n, norm=self.useNormalised, level=self.level)
            self.author_counts[tweet.handle].update(ngrams)
            self.corpus_counts.update(ngrams)
            self.author_ntweets[tweet.handle] += 1
        # self.corpus_ntweets :: number of tweets in whole corpus
        self.corpus_ntweets = sum(self.author_ntweets.values())
        self.authors = set(self.author_ntweets)
        self.nauthors = len(self.authors)

        # normalise counts and prior probabilities
        # corpus-level:
        # self.corpus_nngrams :: total num ngrams seen for all of the corpus
        self.corpus_nngrams = sum(self.corpus_counts.values())
        # self.corpus_probs :: {ngram: prob of ngram for whole corpus}
        self.corpus_probs = Counter(dict_divide(self.corpus_counts, self.corpus_nngrams))
        # self.author_prior :: {author: prob of author tweeting}
        self.author_prior = dict_divide(self.author_ntweets, self.corpus_ntweets)

        # author-level:
        # self.author_probs :: {author: {ngram: prob of ngram for this author}}
        self.author_probs = {}
        # self.author_nngrams :: {author: total num ngrams seen for this author}
        self.author_nngrams = {}
        for author in self.authors:
            profile = self.author_counts[author]
            nngrams = sum(profile.values())
            self.author_probs[author] = self._smoothed_probs(profile, nngrams)
            self.author_nngrams[author] = nngrams

    def predict(self, tweet):
        # featurise the tweet
        ngrams = tweet.ngram(n=self.n, norm=self.useNormalised, level=self.level)
        counts = Counter(ngrams)
        
        # find the nearest author according to negative log-probability distance metric
        best_author, best_distance = "??????", math.inf
        for author in self.authors:
            probs = self.author_probs[author]
            if self.usePriorDist:
                # begin with negative log prior probability
                prior_prob = self.author_prior[author]
                distance = -math.log(prior_prob)
            else:
                distance = 0
            # compute remaining distance as sum of negative log probabilities
            for ngram, count in counts.items():
                # skip ngrams unseen during testing
                # TODO: treat 'UNK' ngrams specially?
                if ngram not in self.corpus_probs:
                    continue
                prob_author_ngram = probs[ngram]
                if prob_author_ngram == 0:
                    distance = math.inf
                    break
                distance -= count * math.log(prob_author_ngram)
            # retain the smallest distance for prediction
            if distance < best_distance:
                best_author, best_distance = author, distance
        
        return best_author

    def _interpret_smoothing(self, smoothing):
        if smoothing.startswith("add"):
            self.smoothing = Smoothing.LAPLACIAN
            self.k = float(smoothing.split("-")[1])
        elif smoothing.startswith("lerp"):
            self.smoothing = Smoothing.INTERPOLATION
            self.alpha = float(smoothing.split("-")[1])
            self.one_minus_alpha = 1 - self.alpha
        elif smoothing.startswith("none"):
            self.smoothing = Smoothing.NONE

    def _smoothed_probs(self, profile, nngrams):
        if self.smoothing == Smoothing.NONE:
            return ProbabilityDistribution(dict_divide(profile, nngrams))
        elif self.smoothing == Smoothing.LAPLACIAN:
            d = len(self.corpus_counts) # number of ngrams ever seen
            smoothed_probs = {ngram: (count + self.k) / (nngrams + d*self.k) for ngram, count in profile.items()}
            default_prob = self.k / (nngrams + d * self.k)
            return DefaultProbabilityDistribution(smoothed_probs, default_prob)
        elif self.smoothing == Smoothing.INTERPOLATION:
            probs = ProbabilityDistribution(dict_divide(profile, nngrams))
            return InterpolatedProbabilityDistribution(self.alpha, probs, self.corpus_probs)


class Smoothing(Enum):
    NONE = 0
    LAPLACIAN = 1
    INTERPOLATION = 2

class ProbabilityDistribution:
    def __init__(self, dist):
        self.dist = dist
    def __getitem__(self, item):
        if item in self.dist:
            return self.dist[item]
        else:
            return 0

class InterpolatedProbabilityDistribution:
    def __init__(self, alpha, dist1, dist2):
        self.alpha = alpha
        self.one_minus_alpha = 1 - alpha
        self.dist1 = dist1
        self.dist2 = dist2
    def __getitem__(self, item):
        prob1 = self.dist1[item]
        prob2 = self.dist2[item]
        return self.one_minus_alpha * prob1 + self.alpha * prob2

class DefaultProbabilityDistribution:
    def __init__(self, dist, default):
        self.dist = dist
        self.default = default
    def __getitem__(self, item):
        if item in self.dist:
            return self.dist[item]
        else:
            return self.default

def dict_divide(d, divisor):
    return {key: val/divisor for key, val in d.items()}

