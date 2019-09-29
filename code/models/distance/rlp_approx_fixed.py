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
        self.author_topL_recentered_normfreqs = {}
        for author, counts in author_counts.items():
            normfreqs = normalise_counter(counts)            

            # recenter the normalised frequences for this author
            recentered_normfreqs = recenter_counter(normfreqs, self.corpus_normfreqs)

            # compute the PROFILE: the L most-distinctive ngrams (largest
            # absolute recentered normalised frquency)
            topL_recentered_normfreqs = truncate_counter(recentered_normfreqs, self.L, key=abs)

            self.author_topL_recentered_normfreqs[author] = topL_recentered_normfreqs

        # construct an inverted index for quickly looping through intersections
        # tqdm.write("inverting author profiles...")
        self.authors_with_ngram = defaultdict(list)
        for author, profile in self.author_topL_recentered_normfreqs.items():
            for ngram in profile.keys():
                self.authors_with_ngram[ngram].append(author)

        # compute author-specific, tweet-independent offsets to distance:
        # NOTE: we will SUBTRACT offset at prediction time, so compute the
        # POSITIVE sum here!
        # tqdm.write("computing offsets...")
        self.author_offset = {}
        for author, profile in self.author_topL_recentered_normfreqs.items():
            offset = 0
            for ngram, RP_a in profile.items():
                offset += self.corpus_normfreqs[ngram] * RP_a
            self.author_offset[author] = offset

        # Also compute normalisation constants, assuming they are
        # also tweet-independent (APPROXIMATION)
        # tqdm.write("computing normalisation terms...")
        self.author_profile_length = {}
        for author, profile in self.author_topL_recentered_normfreqs.items():
            length_squared = 0
            for RP_a in profile.values():
                length_squared += RP_a ** 2
            self.author_profile_length[author] = length_squared ** 0.5

        # # #
        # BEGIN ADDED CODE

        # WAIT! We need to compute that bottom sum after all!
        # and we need sum_x_in_Xa E(x)^2 to do it!
        self.tweet_length_sum_part_2 = {}
        for author, profile in self.author_topL_recentered_normfreqs.items():
            sum_part_2 = 0
            for ngram in profile.keys():
                sum_part_2 += self.corpus_normfreqs[ngram] ** 2
            self.tweet_length_sum_part_2[author] = sum_part_2

        # END ADDED CODE
        # # #

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
                RP_a = self.author_topL_recentered_normfreqs[author][ngram]
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
            if profile_length:
                similarity[author] = numerator[author] / profile_length

        # # #
        # BEGIN ADDED CODE

        # ACTUALLY, we do need the ||RP_t|| term, because there is some
        # author dependence I didn't notice the first time through
        # compute part 1 (specific to tweet)
        tweet_length_sum_part_1 = 0
        for ngram, RP_t in tweet_recentered_normfreqs.items():
            tweet_length_sum_part_1 += RP_t ** 2
        # we pre-computed part 2 for each author at training time
        # compute part 3 for each author now using index:
        tweet_length_sum_part_3 = defaultdict(float)
        for ngram in tweet_recentered_normfreqs.keys():
            E_squared = self.corpus_normfreqs[ngram] ** 2
            for author in self.authors_with_ngram[ngram]:
                tweet_length_sum_part_3[author] += E_squared
        # with all three parts, we can compute the tweet lengths, and normalise
        # the similarities using its sqrt:
        for author, sum_part_3 in tweet_length_sum_part_3.items():
            tweet_length_squared = tweet_length_sum_part_1 + self.tweet_length_sum_part_2[author] - sum_part_3
            tweet_length = tweet_length_squared ** 0.5
            similarity[author] /= tweet_length

        # END ADDED CODE
        # # #

        # we could compute the distance as 1-similarity, or just maximise
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

def truncate_counter(counter, L, key=id):
    top_L_items = nlargest(L, counter.items(), key=lambda item: key(item[1]))
    return Counter(dict(top_L_items))
