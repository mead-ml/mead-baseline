import numpy as np
from baseline.utils import (
    load_user_classifier_model, create_user_classifier_model,
    load_user_tagger_model, create_user_tagger_model,
    load_user_seq2seq_model, create_user_seq2seq_model,
    load_user_lang_model, create_user_lang_model,
    load_user_embeddings,
    lowercase, revlut,
    export, wrapped_partial, is_sequence
)
from baseline.vectorizers import Token1DVectorizer, Char2DVectorizer, Dict1DVectorizer, Dict2DVectorizer

__all__ = []
exporter = export(__all__)


@exporter
class Classifier(object):
    """Text classifier
    
    Provide an interface to DNN classifiers that use word lookup tables.
    """
    def __init__(self):
        super(Classifier, self).__init__()

    def save(self, basename):
        """Save this model out
             
        :param basename: Name of the model, not including suffixes
        :return: None
        """
        pass

    @classmethod
    def load(cls, basename, **kwargs):
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

    def get_vocab(self, name='word'):
        """Return the vocabulary, which is a dictionary mapping a word to its word index
        
        :return: A dictionary mapping a word to its word index
        """
        pass

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.
        
        :return: A list of the labels for the decision
        """
        pass

    def classify_text(self, tokens, **kwargs):
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
        if type(tokens[0]) == str:
            tokens_seq = (tokens,)
            max_length = len(tokens)
        else:
            tokens_seq = tokens
            max_length = 0
            for t in tokens_seq:
                max_length = max(max_length, len(t))

        vectorizers = kwargs.get('vectorizers')
        if vectorizers is None:
            vectorizers = {'word': Token1DVectorizer(mxlen=kwargs.get('mxlen', max_length))}

        examples = dict()
        for k in vectorizers.keys():
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in vectorizers.items():
                vec, length = vectorizer.run(tokens, self.embeddings[k].vocab)
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in vectorizers.keys():
            examples[k] = np.stack(examples[k])
        outcomes_list = self.classify(examples)
        results = []
        for outcomes in outcomes_list:
            results += [sorted(outcomes, key=lambda tup: tup[1], reverse=True)]
        return results


@exporter
def create_model(known_creators, input_, output_, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default

    For classification and tagging tasks input_ is embeddings and output_ is labels
    For seq2seq tasks input_ is the source embeddings and output_ is the target embeddings

    :param known_creators: Map of baseline creators, keyed by `model_type`, typically a static factory method
    :param input_: parameter for the input, general word vectors
    :param output_: parameter for the output, generally labels or output vectors
    :param kwargs: Anything required to feed the model its parameters
    :return: A newly created model
    """
    model_type = kwargs.get('model_type', 'default')
    creator_fn = known_creators[model_type] if model_type in known_creators else kwargs['task_fn']
    print('Calling model ', creator_fn)
    return creator_fn(input_, output_, **kwargs)

create_classifier_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_classifier_model,
        name='create_classifier_model'
    )
)
create_tagger_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_tagger_model,
        name='create_tagger_model'
    )
)
create_seq2seq_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_seq2seq_model,
        name='create_seq2seq_model'
    )
)


@exporter
def create_lang_model(known_creators, embeddings, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_creators:
        creator_fn = known_creators[model_type]
        print('Calling baseline model creator ', creator_fn)
        return creator_fn(embeddings, **kwargs)
    return create_user_lang_model(embeddings, **kwargs)


@exporter
def load_model(known_loaders, outname, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default

    :param known_loaders: Map of baseline functions to load the model, typically a static factory method
    :param outname The model name to load
    :param kwargs: Anything required to feed the model its parameters
    :return: A restored model
    """
    model_type = kwargs.get('model_type', 'default')
    loader_fn = known_loaders[model_type] if model_type in known_loaders else kwargs['task_fn']
    return loader_fn(outname, **kwargs)

load_classifier_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_classifier_model,
        name='load_classifier_model'
    )
)
load_tagger_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_tagger_model,
        name='load_tagger_model'
    )
)
load_seq2seq_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_seq2seq_model,
        name='load_seq2seq_model'
    )
)
load_lang_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_lang_model,
        name='load_seq2seq_model'
    )
)


@exporter
class Tagger(object):
    """Structured prediction classifier, AKA a tagger
    
    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
    def __init__(self):
        super(Tagger, self).__init__()

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

    def predict_text(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        label_field = kwargs.get('label', 'label')

        mxlen = 0
        mxwlen = 0
        if type(tokens[0]) == str:
            mxlen = len(tokens)
            tokens_seq = []
            for t in tokens:
                mxwlen = max(mxwlen, len(t))
                tokens_seq += [dict({'text': t})]
            tokens_seq = [tokens_seq]
        else:
            # Better be a sequence, but it could be pre-batched, [[],[]]
            # But what kind of object is at the first one then?
            if is_sequence(tokens[0]):
                tokens_seq = []
                # Then what we have is [['The', 'dog',...], ['I', 'cannot']]
                # For each of the utterances, we need to make a dictionary
                if type(tokens[0][0]) == str:

                    for utt in tokens:
                        utt_dict_seq = []
                        mxlen = max(mxlen, len(utt))
                        for t in utt:
                            mxwlen = max(mxwlen, len(t))
                            utt_dict_seq += [dict({'text': t})]
                        tokens_seq += [utt_dict_seq]
                # Its already in dict form so we dont need to do anything
                elif type(tokens[0][0]) == dict:
                    for utt in tokens:
                        mxlen = max(mxlen, len(utt))
                        for t in utt['text']:
                            mxwlen = max(mxwlen, len(t))
            # If its a dict, we just wrap it up
            elif type(tokens[0]) == dict:
                mxlen = max(len(tokens))
                for t in tokens:
                    mxwlen = max(mxwlen, len(t))
                tokens_seq = [tokens]
            else:
                raise Exception('Unknown input format')

        if len(tokens_seq) == 0:
            return []

        vectorizers = kwargs.get('vectorizers')
        if vectorizers is None:
            word_tokenizer = Dict1DVectorizer(mxlen=mxlen)
            char_tokenizer = Dict2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
            vectorizers = {'word': word_tokenizer, 'char': char_tokenizer}

        # This might be inefficient if the label space is large

        label_vocab = revlut(self.get_labels())

        examples = dict()
        for k, vectorizer in vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in vectorizers.items():
                vec, length = vectorizer.run(tokens, self.embeddings[k].vocab)
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in vectorizers.keys():
            examples[k] = np.stack(examples[k])

        outcomes = self.predict(examples)
        outputs = []
        for i, outcome in enumerate(outcomes):
            output = []
            for j, token in enumerate(tokens_seq[i]):
                new_token = dict()
                new_token.update(token)
                new_token[label_field] = label_vocab[outcome[j].item()]
                output += [new_token]
            outputs += [output]
        return outputs

    def get_vocab(self, vocab_type='word'):
        pass

    def get_labels(self):
        pass


@exporter
class LanguageModel(object):

    def __init__(self):
        super(LanguageModel, self).__init__()

    def step(self, batch_time, context):
        pass


@exporter
class EncoderDecoder(object):

    def save(self, model_base):
        pass

    def __init__(self):
        super(EncoderDecoder, self).__init__()

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

