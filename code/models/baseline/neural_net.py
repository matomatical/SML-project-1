from sklearn.neural_network import MLPClassifier
from collections import defaultdict


# def _ddictpickle(): # needed to pickle the module
#     return defaultdict(int)

class Model:
    # TODO change to use pytorch instead
    def __init__(self, data, n, w, hidden_dim, seed):
        # not sure how to model size of hidden_dimension
        self.cgram_len = int(n)
        self.wgram_len = int(w)
        self.hidden    = int(hidden_dim)
        self.seed      = int(seed)
        self.data      = data # save for future use

        self.indices = defaultdict(int) # {handle: {ngram: count, ...}, ...}
        self.handles = defaultdict(int)
        # After triming converted to {handle: set(top_L_ngrams)}

        self.ngrams = defaultdict(int) # ngram: count, for removing single instance ngrams

        # self.invertedNgram = defaultdict(set) # {ngram: set(handles), ...} used for inverted index
        self.total_grams = 0
        self.total_handles = 0
        for t in data:
            # count ngrams
            for ng in t.char_ngram(self.cgram_len):
                self.ngrams[ng] += 1

            for wg in t.word_ngram(self.wgram_len):
                self.ngrams[wg] += 1
                
            # record handles and their indices
            if t.handle not in self.handles:
                self.handles[t.handle] = self.total_handles
                self.total_handles += 1

        discarded = 0
        for gram, count in self.ngrams.items():
            if count > 1:
                self.indices[gram] = self.total_grams
                self.total_grams += 1
            else:
                discarded += 1
            # record ngrams and their indices
            # for ng in t.char_ngram(self.cgram_len): # 19030707 grams
            #     self.indices[ng] = self.total_grams
            #     self.total_grams += 1
            # for wg in t.word_ngram(self.wgram_len): # 4674715 grams
            #     self.indices[wg] = self.total_grams
            #     self.total_grams += 1


        # without trimming ~3,700,000 ngrams x ~260,000 = 15.1 terabytes, assuming 64 bit ints
        # with trimming      ~270,000 ngrams x ~260,000 =  1.1 terabytes, also 64 bit ints
        # unfortunately, generator doesn't work because MLPClassifier assumes it's a scalar array and it needs a 2d array
        print(f"begin training: {self.total_grams} grams x {len(self.data)} instances") 
        print(f"discarded: {discarded} grams")
        # self.classifier = MLPClassifier(activation="logistic", hidden_layer_sizes=(self.hidden,), random_state=self.seed)

        # classifier_in = [x for x in self.generative_inputs()]
        # classifier_classes = [y for y in self.generative_labels()]

        # self.classifier.fit(classifier_in, classifier_classes)

    def ngram_vector(self, t):
        vec = [0] * self.total_grams
        for ng in t.char_ngram(self.cgram_len):
            i = self.indices[ng]
            vec[i] = 1
        for wg in t.word_ngram(self.wgram_len):
            i = self.indices[wg]
            vec[i] = 1
        return vec

    def handle_vector(self, t):
        vec = [0] * self.total_handles
        i = self.handles[t.handle]
        vec[i] = 1

        return vec

    def generative_inputs(self):
        for t in self.data:
            yield self.ngram_vector(t)
        

    def generative_labels(self):
        for t in self.data:
            # yield self.handle_vector(t)
            yield t.handle

    def predict(self, t):
        prediction = self.classifier.predict([self.ngram_vector(t)])[0]

        return 0
        