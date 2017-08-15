import numpy as np
from baseline.utils import revlut, load_user_model, create_user_model


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


def create_classifier_model(default_create_model_fn, w2v, labels, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        return default_create_model_fn(w2v, labels, **kwargs)

    model = create_user_model(w2v, labels, **kwargs)
    return model


def load_classifier_model(default_load_fn, outname, **kwargs):

    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        print('Calling default load fn', default_load_fn)
        return default_load_fn(outname, **kwargs)
    return load_user_model(outname, **kwargs)


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

    def predict_text(self, tokens, mxlen, maxw, zero_alloc=np.zeros):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed
        :param tokens: 
        :param mxlen: 
        :param maxw: 
        :param zero_alloc: 
        :return: 
        """
        words_vocab = self.get_vocab(vocab_type='word')
        chars_vocab = self.get_vocab(vocab_type='char')
        # This might be inefficient if the label space is large
        label_vocab = revlut(self.get_labels())
        xs = zero_alloc((1, mxlen), dtype=int)
        xs_ch = zero_alloc((1, mxlen, maxw), dtype=int)
        lengths = zero_alloc(1, dtype=int)
        lengths[0] = min(len(tokens), mxlen)
        for j in range(mxlen):

            if j == len(tokens):
                break

            w = tokens[j]
            nch = min(len(w), maxw)

            xs[0, j] = words_vocab.get(w, 0)
            for k in range(nch):
                xs_ch[0, j, k] = chars_vocab.get(w[k], 0)

        indices = self.predict(xs, xs_ch, lengths)[0]
        output = []
        for j in range(lengths[0]):
            output.append((tokens[j], label_vocab[indices[j]]))
        return output

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

    def get_src_vocab(self):
        pass

    def get_dst_vocab(self):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass
