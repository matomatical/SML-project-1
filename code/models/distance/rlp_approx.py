from heapq import nlargest
from collections import defaultdict, Counter

from tqdm import tqdm

class Model:
    def __init__(self, data, n, level, L, norm, truncate="False"):
        self.L = int(L)
        self.n = int(n)
        self.level = level
        self.norm = eval(norm)
        self.truncate = eval(truncate)

        # compute corpus and author (unnormalised) frequencies:
        # tqdm.write("counting ngrams from input tweets...")
        corpus_counts = Counter()
        author_counts = defaultdict(Counter)
        # we'll count the number of tweets per author while we're at it
        self.author_numtweets = defaultdict(int)
        for tweet in data:
            tweet_ngrams = self.ngrams(tweet)
            corpus_counts.update(tweet_ngrams)
            author_counts[tweet.handle].update(tweet_ngrams)
            self.author_numtweets[tweet.handle] += 1

        # normalise these counts into normalised frequencies/probabilities
        # and recenter them to each author's profile
        # tqdm.write("normalising and recentering ngram frequencies...")
        self.corpus_normfreqs = normalise_counter(corpus_counts)
        self.author_recentered_normfreqs = {}
        self.author_top_L_ngrams = {}
        for author, counts in author_counts.items():
            normfreqs = normalise_counter(counts)            

            # recenter the normalised frequences for this author
            recentered_normfreqs = recenter_counter(normfreqs, self.corpus_normfreqs)

            # compute the PROFILE: the L most-distinctive ngrams (largest absolute recentered normalised frquency)
            ngrams = recentered_normfreqs.keys()
            top_L_ngrams = nlargest(self.L, ngrams, key=lambda ngram: abs(recentered_normfreqs[ngram]))

            self.author_recentered_normfreqs[author] = recentered_normfreqs
            self.author_top_L_ngrams[author] = top_L_ngrams

        # construct an inverted index for quickly looping through intersections
        # tqdm.write("inverting author profiles...")
        self.authors_with_ngram = defaultdict(list)
        for author, top_L_ngrams in self.author_top_L_ngrams.items():
            for ngram in top_L_ngrams:
                self.authors_with_ngram[ngram].append(author)

        # compute author-specific, tweet-independent offsets to distance:
        # NOTE: we will SUBTRACT offset at prediction time, so compute the
        # POSITIVE sum here!
        # tqdm.write("computing offsets...")
        self.author_offset = {}
        for author, top_L_ngrams in self.author_top_L_ngrams.items():
            offset = 0
            for ngram in top_L_ngrams:
                offset += self.corpus_normfreqs[ngram] * self.author_recentered_normfreqs[author][ngram]
            self.author_offset[author] = offset

        # Also compute normalisation constants, assuming they are
        # also tweet-independent (APPROXIMATION)
        # tqdm.write("computing normalisation terms...")
        self.author_profile_length = {}
        for author, top_L_ngrams in self.author_top_L_ngrams.items():
            length_squared = 0
            for ngram in top_L_ngrams:
                length_squared += self.author_recentered_normfreqs[author][ngram] ** 2
            self.author_profile_length[author] = length_squared ** 0.5

        # done!

    def predict(self, tweet):
        tweet_ngrams = self.ngrams(tweet)
        tweet_normfreqs = normalise_counter(Counter(tweet_ngrams))
        tweet_recentered_normfreqs = recenter_counter(tweet_normfreqs, self.corpus_normfreqs)

        # first, compute the numerators for each author:
        # (APPROXIMATE WITH INTERSECTION - OFFSET)
        numerator = defaultdict(float)
        for ngram, RP_t in tweet_recentered_normfreqs.items():
            E = self.corpus_normfreqs[ngram]
            for author in self.authors_with_ngram[ngram]:
                RP_a = self.author_recentered_normfreqs[author][ngram]
                numerator[author] += RP_a * RP_t + RP_a * E
                if self.truncate:
                    numerator[author] += RP_t * E
        for author, offset in self.author_offset.items():
            numerator[author] -= offset

        # we need to normalise these numerators by ||RP_a|| * ||RP_t||
        # (ignore ||RP_t|| because it's a constant for all authors)
        # (APPROXIMATE ||RP_a|| as precomputed normalisation factor from
        # only those ngrams in the top L)
        similarity = {}
        for author, profile_length in self.author_profile_length.items():
            similarity[author] = numerator[author] / profile_length

        # we could compute the distance as 1-simularity, or just maximise
        # similarity now, to make our prediction:
        return max(similarity.items(), key=lambda a_s: a_s[1])[0]
        

    # helper function to get the right ngrams
    def ngrams(self, tweet):
        return list(tweet.ngram(n=self.n, level=self.level, norm=self.norm))

def normalise_counter(counter):
    total = sum(counter.values())
    return Counter({k: v/total for k, v in counter.items()})

def recenter_counter(counter, center):
    return Counter({k: v-center[k] for k, v in counter.items()})
