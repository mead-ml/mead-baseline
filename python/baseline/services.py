import numpy as np
from baseline.utils import (export,
                            unzip_files,
                            find_model_basename,
                            find_files_with_prefix,
                            read_json,
                            is_sequence,
                            revlut,
                            load_vectorizers,
                            load_vocabs)

import os
import pickle

__all__ = []
exporter = export(__all__)


@exporter
class ClassifierService(object):

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.model.get_labels()

    @classmethod
    def load(cls, bundle, **kwargs):
        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        model_basename = find_model_basename(directory)
        if model_basename.find('-tf-') >= 0:
            import baseline.tf.classify as classify
        elif model_basename.find('-keras-') >= 0:
            import baseline.keras.classify as classify
        elif model_basename.endswith(".pyt"):
            import baseline.pytorch.classify as classify
        else:
            import baseline.dy.classify as classify
        model = classify.load_model(model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

    def transform(self, tokens):
        """Take tokens and apply the internal vocab and vectorizers.  The tokens should be either a batch of text
        single utterance of type ``list``
        """
        mxlen = 0
        mxwlen = 0
        if type(tokens[0]) == str:
            tokens_seq = (tokens,)
        else:
            tokens_seq = tokens

        for tokens in tokens_seq:
            mxlen = max(mxlen, len(tokens))
            for token in tokens:
                mxwlen = max(mxwlen, len(token))

        examples = dict()
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
        outcomes_list = self.model.classify(examples)
        results = []
        for outcomes in outcomes_list:
            results += [sorted(outcomes, key=lambda tup: tup[1], reverse=True)]
        return results


@exporter
class TaggerService(object):

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.model.get_labels()

    @classmethod
    def load(cls, bundle, **kwargs):
        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        model_basename = find_model_basename(directory)
        if model_basename.find('-tf-') >= 0:
            import baseline.tf.tagger as tagger
        elif model_basename.endswith(".pyt"):
            import baseline.pytorch.tagger as tagger
        else:
            import baseline.dy.tagger as tagger
        model = tagger.load_model(model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

    def transform(self, tokens, **kwargs):
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
                # [[{'text': 'The', 'pos': 'DT'}, ...

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

        # This might be inefficient if the label space is large

        label_vocab = revlut(self.get_labels())

        examples = dict()
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])

        outcomes = self.model.predict(examples)
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


@exporter
class LanguageModelService(object):

    def __init__(self):
        super(LanguageModelService, self).__init__()

    def step(self, batch_time, context):
        pass


@exporter
class EncoderDecoderService(object):

    def save(self, model_base):
        pass

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        self.vectorizers = vectorizers
        self.model = model
        self.src_vocabs = {}
        self.dst_vocab = None
        for k, vocab in vocabs.items():
            if k == 'dst':
                self.dst_vocab = vocab
            else:
                self.src_vocabs[k] = vocab

        self.dst_idx_to_token = revlut(self.dst_vocab)
        self.src_vectorizers = {}
        self.dst_vectorizer = None
        for k, vectorizer, in vocabs.items():
            if k == 'dst':
                self.dst_vectorizer = vectorizer
            else:
                self.src_vectorizers[k] = vectorizer

    def get_src_vocab(self, vocab_type):
        return self.src_vocabs[vocab_type]

    def get_dst_vocab(self):
        return self.dst_vocab

    @classmethod
    def load(cls, bundle, **kwargs):

        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)
        model_basename = find_model_basename(directory)
        if model_basename.find('-tf-') >= 0:
            import baseline.tf.seq2seq as seq2seq
        elif model_basename.endswith(".pyt"):
            import baseline.pytorch.seq2seq as seq2seq
        else:
            import baseline.dy.seq2seq as seq2seq

        model = seq2seq.load_model(model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

    def transform(self, tokens, **kwargs):
        source_dict = {}
        for k, vectorizer in self.vectorizers:
            source_dict[k], lengths = vectorizer.run(tokens, self.src_vocabs[k], kwargs.get('beam', 1))
            if lengths is not None:
                source_dict['{}_lengths'.format(k)]

        z = self.model.run(source_dict)[0]
        best = z[0]
        out = []
        for i in range(len(best)):
            word = self.dst_idx_to_token.get(best[i], '<PAD>')
            if word != '<PAD>' and word != '<EOS>':
                out.append(word)
        return out

