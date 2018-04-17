import numpy as np
from baseline.utils import revlut, load_user_classifier_model, create_user_classifier_model, load_user_tagger_model, create_user_tagger_model
from baseline.utils import load_user_seq2seq_model, create_user_seq2seq_model, create_user_lang_model, lowercase


class Classifier(object):
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

    def classify(self, batch_dict):
        """Classify a batch of text with whatever features the model can use from the batch_dict.
        The indices correspond to get_vocab().get('word', 0)
        
        :param batch_dict: This normally contains `x`, a `BxT` tensor of indices.  Some classifiers and readers may
        provide other features

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

    def classify_text(self, tokens, mxlen, zero_alloc=np.zeros, word_trans_fn=lowercase):
        """Utility method to convert a list of words comprising a text to indices, and create a single element
        batch which is then classified.  The returned decision is sorted in descending order of probability.

        At the moment, this method only prepares `x` features in the `batch_dict`.  This means that it cannot
        be used for models that provide, for instance, character-level features.  In that case, use `classify` directly.
        
        :param tokens: A list of words
        :param mxlen: The maximum length of the words.  List items beyond this edge are removed
        :param zero_alloc: A function defining an allocator.  Defaults to numpy zeros
        :param word_trans_fn: A transform on the input word
        :return: A sorted list of outcomes for a single element batch
        """
        vocab = self.get_vocab()
        x = zero_alloc((1, mxlen), dtype=int)
        length = min(len(tokens), mxlen)
        for j in range(length):
            word = word_trans_fn(tokens[j])
            if word not in vocab:
                if word != '':
                    print(word)
                    idx = 0
            else:
                idx = vocab[word]
            x[0, j] = idx
        outcomes = self.classify({'x': x})[0]
        return sorted(outcomes, key=lambda tup: tup[1], reverse=True)


def create_classifier_model(known_creators, w2v, labels, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default
    
    :param known_creators: Map of baseline creators, keyed by `model_type`, typically a static factory method
    :param w2v: Word embeddings
    :param labels: A list or map of labels
    :param kwargs: Anything required to feed the model its parameters
    :return: A newly created model
    """
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_creators:
        creator_fn = known_creators[model_type]
        print('Calling baseline model ', creator_fn)
        return creator_fn(w2v, labels, **kwargs)

    model = create_user_classifier_model(w2v, labels, **kwargs)
    return model


def load_classifier_model(known_loaders, outname, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default
    
    :param known_loaders: Map of baseline functions to load the model, typically a static factory method
    :param outname The model name to load
    :param kwargs: Anything required to feed the model its parameters
    :return: A restored model
    """

    model_type = kwargs.get('model_type', 'default')
    if model_type in known_loaders:
        loader_fn = known_loaders[model_type]
        print('Calling baseline model loader ', loader_fn)
        return loader_fn(outname, **kwargs)
    return load_user_classifier_model(outname, **kwargs)


class Tagger(object):
    """Structured prediction classifier, AKA a tagger
    
    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
    def __init__(self):
        pass

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

    def predict_text(self, tokens, mxlen, maxw, zero_alloc=np.zeros, word_trans_fn=lowercase):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: 
        :param mxlen: 
        :param maxw: 
        :param zero_alloc: Define
        :param word_trans_fn:
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

            xs[0, j] = words_vocab.get(word_trans_fn(w), 0)
            for k in range(nch):
                xs_ch[0, j, k] = chars_vocab.get(w[k], 0)

        indices = self.predict({'x': xs, 'xch': xs_ch, 'lengths': lengths})[0]
        output = []
        for j in range(lengths[0]):
            output.append((tokens[j], label_vocab[indices[j]]))
        return output

    def get_vocab(self, vocab_type='word'):
        pass

    def get_labels(self):
        pass


def create_tagger_model(default_create_model_fn, labels, embeddings, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        return default_create_model_fn(labels, embeddings, **kwargs)

    model = create_user_tagger_model(labels, embeddings, **kwargs)
    return model


def load_tagger_model(default_load_fn, outname, **kwargs):

    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        print('Calling default load fn', default_load_fn)
        return default_load_fn(outname, **kwargs)
    return load_user_tagger_model(outname, **kwargs)


class LanguageModel(object):

    def __init__(self):
        pass

    def step(self, batch_time, context):
        pass


class EncoderDecoder(object):

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

    def run(self, source_dict, **kwargs):
        pass


def create_seq2seq_model(known_creators, input_embedding, output_embedding, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_creators:
        creator_fn = known_creators[model_type]
        return creator_fn(input_embedding, output_embedding, **kwargs)

    model = create_user_seq2seq_model(input_embedding, output_embedding, **kwargs)
    return model


def load_seq2seq_model(known_loaders, outname, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_loaders:
        loader_fn = known_loaders[model_type]
        return loader_fn(outname, **kwargs)
    return load_user_seq2seq_model(outname, **kwargs)


def create_lang_model(known_creators, embeddings, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_creators:
        creator_fn = known_creators[model_type]
        print('Calling baseline model loader ', creator_fn)
        return creator_fn(embeddings, **kwargs)
    return create_user_lang_model(embeddings, **kwargs)

