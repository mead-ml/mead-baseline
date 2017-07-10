import numpy as np


class Classifier:
    """Text classifier
    
    Provide an interface to DNN classifiers that use word lookup tables.
    """
    def __init__(self):
        pass

    def save(self, basename):
        """Save this model out
             
        :param basename: Name of the model, not including suffixes
        :return: None
        """
        pass

    @staticmethod
    def load(basename, **kwargs):
        """Load the model from a basename, including directory
        
        :param basename: Name of the model, not including suffixes
        :param kwargs: Anything that is useful to optimize experience for a specific framework
        :return: A newly created model
        """
        pass

    def classify(self, batch_time):
        """Classify a batch of text as tensor of BxT indices to words.
        The indices correspond to get_vocab().get('word', 0)
        
        :param batch_time: BxT tensor of indices
        :return: A list of lists of tuples (label, value)
        """
        pass

    def get_vocab(self):
        """Return the vocabulary, which is a dictionary mapping a word to its word index
        
        :return: A dictionary mapping a word to its word index
        """
        pass

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.
        
        :return: A list of the labels for the decision
        """
        pass

    def classify_text(self, tokens, mxlen, zeropad=0, zero_alloc=np.zeros):
        """Utility method to convert a list of words comprising a text to indices, and create a single element
        batch which is then classified.  The returned decision is sorted in descending order of probability
        
        :param tokens: A list of words
        :param mxlen: The maximum length of the words.  List items beyond this edge are removed
        :param zeropad: How much zero-padding (total) to allocate the signal
        :param zero_alloc: A function defining an allocator.  Defaults to numpy zeros
        :return: A sorted list of outcomes for a single element batch
        """
        vocab = self.get_vocab()
        x = zero_alloc((1, mxlen), dtype=int)
        halffiltsz = zeropad // 2
        length = min(len(tokens), mxlen - zeropad + 1)
        for j in range(length):
            word = tokens[j]
            if word not in vocab:
                if word != '':
                    print(word)
                    idx = 0
            else:
                idx = vocab[word]
            x[0, j + halffiltsz] = idx
        outcomes = self.classify(x)[0]
        return sorted(outcomes, key=lambda tup: tup[1], reverse=True)


class Tagger:

    def __init__(self):
        pass

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, x, xch, lengths):
        pass

    def predict_text(self, tokens, mxlen, zeropad=0, zero_alloc=np.zeros):
        pass

    def get_vocab(self, vocab_type='word'):
        pass

    def get_labels(self):
        pass


class LanguageModel:

    def __init__(self):
        pass

    def step(self, batch_time, context):
        pass


class EncoderDecoder:

    def save(self, model_base):
        pass

    def __init__(self):
        pass

    @staticmethod
    def create(src_vocab, dst_vocab, **kwargs):
        pass

    def create_loss(self):
        pass