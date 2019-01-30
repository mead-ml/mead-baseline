import six

import os
import pickle
from collections import defaultdict
import numpy as np
import baseline
from baseline.utils import (
    export,
    unzip_files,
    find_model_basename,
    find_files_with_prefix,
    import_user_module,
    read_json,
    is_sequence,
    revlut,
    load_vectorizers,
    load_vocabs,
    lookup_sentence,
    normalize_backend,
)
from baseline.model import load_model_for


__all__ = []
exporter = export(__all__)


class Service(object):

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.model.get_labels()

    @classmethod
    def signature_name(cls):
        raise Exception("Undefined signature name")

    @classmethod
    def task_name(cls):
        raise Exception("Undefined task name")

    def batch_input(self, tokens):
        """Turn the input into a consistent format.

        :return: List[List[str]]
        """
        mxlen = 0
        mxwlen = 0
        # If the input is List[str] wrap it in list to make a batch of size one.
        tokens_seq = (tokens,) if isinstance(tokens[0], six.string_types) else tokens
        # Get sentence and word lengths from the batch
        for tokens in tokens_seq:
            mxlen = max(mxlen, len(tokens))
            for token in tokens:
                mxwlen = max(mxwlen, len(token))
        return tokens_seq, mxlen, mxwlen

    def set_vectorizer_lens(self, mxlen, mxwlen):
        """Set the max lengths on the vectorizers if unset.

        :param mxlen: `int`: The max length of an example
        :param mxwlen: `int`: The max length of a word in the batch
        """
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen

    def vectorize(self, tokens_seq):
        """Turn the input into that batch dict for prediction.

        :param tokens_seq: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    examples[lengths_key].append(length)

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
            lengths_key = '{}_lengths'.format(k)
            if lengths_key in examples:
                examples[lengths_key] = np.stack(examples[lengths_key])
        return examples

    @classmethod
    def load(cls, bundle, **kwargs):
        """Load a model from a bundle.

        This can be either a local model or a remote, exported model.

        :returns a Service implementation
        """
        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory)
        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        be = normalize_backend(kwargs.get('backend', 'tf'))

        remote = kwargs.get("remote", None)
        name = kwargs.get("name", None)
        if remote:
            beam = kwargs.get('beam', 10)
            model = Service._create_remote_model(directory, be, remote, name, cls.signature_name(), beam, preproc=kwargs.get('preproc', False))
            return cls(vocabs, vectorizers, model)

        # Currently nothing to do here
        # labels = read_json(os.path.join(directory, model_basename) + '.labels')

        import_user_module('baseline.{}.embeddings'.format(be))
        import_user_module('baseline.{}.{}'.format(be, cls.task_name()))
        model = load_model_for(cls.task_name(), model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

    @staticmethod
    def _create_remote_model(directory, backend, remote, name, signature_name, beam, preproc='client'):
        """Reads the necessary information from the remote bundle to instatiate
        a client for a remote model.

        :directory the location of the exported model bundle
        :remote a url endpoint to hit
        :name the model name, as defined in tf-serving's model.config
        :signature_name  the signature to use.
        :beam used for s2s and found in the kwargs. We default this and pass it in.

        :returns a RemoteModel
        """
        assets = read_json(os.path.join(directory, 'model.assets'))
        model_name = assets['metadata']['exported_model']
        labels = read_json(os.path.join(directory, model_name) + '.labels')
        lengths_key = assets.get('lengths_key', None)
        inputs = assets.get('inputs', [])

        if backend == 'tf':
            remote_models = import_user_module('baseline.remote')
            if remote.startswith('http'):
                RemoteModel = remote_models.RemoteModelTensorFlowREST
            elif preproc == 'server':
                RemoteModel = remote_models.RemoteModelTensorFlowGRPCPreproc
            else:
                RemoteModel = remote_models.RemoteModelTensorFlowGRPC
            model = RemoteModel(remote, name, signature_name, labels=labels, lengths_key=lengths_key, inputs=inputs, beam=beam)
        else:
            raise ValueError("only Tensorflow is currently supported for remote Services")

        return model


@exporter
class ClassifierService(Service):

    @classmethod
    def task_name(cls):
        return 'classify'

    @classmethod
    def signature_name(cls):
        return 'predict_text'

    def predict(self, tokens, preproc='client'):
        """Take tokens and apply the internal vocab and vectorizers.  The tokens should be either a batch of text
        single utterance of type ``list``
        """
        token_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        examples = self.vectorize(token_seq)
        if preproc == 'server':
            examples['tokens'] = [" ".join(x) for x in token_seq]
        outcomes_list = self.model.predict(examples)
        results = []
        for outcomes in outcomes_list:
            results += [list(map(lambda x: (x[0], x[1].item()), sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
        return results


@exporter
class TaggerService(Service):

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        super(TaggerService, self).__init__(vocabs, vectorizers, model)
        self.label_vocab = revlut(self.get_labels())

    @classmethod
    def task_name(cls):
        return 'tagger'

    @classmethod
    def signature_name(cls):
        return 'tag_text'

    def batch_input(self, tokens):
        """Convert the input into a consistent format.

        :return: List[List[dict[str] -> str]]
        """
        mxlen = 0
        mxwlen = 0
        # Input is a list of strings. (assume strings are tokens)
        if isinstance(tokens[0], six.string_types):
            mxlen = len(tokens)
            tokens_seq = []
            for t in tokens:
                mxwlen = max(mxwlen, len(t))
                tokens_seq.append({'text': t})
            tokens_seq = [tokens_seq]
        else:
            # Better be a sequence, but it could be pre-batched, [[],[]]
            # But what kind of object is at the first one then?
            if is_sequence(tokens[0]):
                tokens_seq = []
                # Then what we have is [['The', 'dog',...], ['I', 'cannot']]
                # [[{'text': 'The', 'pos': 'DT'}, ...

                # For each of the utterances, we need to make a dictionary
                if isinstance(tokens[0][0], six.string_types):
                    for utt in tokens:
                        utt_dict_seq = []
                        mxlen = max(mxlen, len(utt))
                        for t in utt:
                            mxwlen = max(mxwlen, len(t))
                            utt_dict_seq += [dict({'text': t})]
                        tokens_seq += [utt_dict_seq]
                # Its already in dict form so we dont need to do anything
                elif isinstance(tokens[0][0], dict):
                    for utt in tokens:
                        mxlen = max(mxlen, len(utt))
                        for t in utt['text']:
                            mxwlen = max(mxwlen, len(t))
            # If its a dict, we just wrap it up
            elif isinstance(tokens[0], dict):
                mxlen = max(len(tokens))
                for t in tokens:
                    mxwlen = max(mxwlen, len(t))
                tokens_seq = [tokens]
            else:
                raise Exception('Unknown input format')

        if len(tokens_seq) == 0:
            return []
        return tokens_seq, mxlen, mxwlen

    def predict(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        preproc = kwargs.get('preproc', 'client')
        label_field = kwargs.get('label', 'label')
        tokens_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        examples = self.vectorize(tokens_seq)
        if preproc == 'server':
            examples['tokens'] = [" ".join([y['text'] for y in x]) for x in tokens_seq]

        outcomes = self.model.predict(examples)
        outputs = []
        for i, outcome in enumerate(outcomes):
            output = []
            for j, token in enumerate(tokens_seq[i]):
                new_token = dict()
                new_token.update(token)
                new_token[label_field] = self.label_vocab[outcome[j].item()]
                output += [new_token]
            outputs += [output]
        return outputs


@exporter
class LanguageModelService(Service):

    @classmethod
    def task_name(cls):
        return "lm"

    def __init__(self, *args, **kwargs):
        super(LanguageModelService, self).__init__(*args, **kwargs)
        self.idx_to_token = revlut(self.vocabs[self.model.tgt_key])

    @classmethod
    def load(cls, bundle, **kwargs):
        kwargs['batchsz'] = 1
        return super(LanguageModelService, cls).load(bundle, **kwargs)

    # Do a greedy decode for now, everything else will be super slow
    def predict(self, tokens, **kwargs):
        mxlen = kwargs.get('mxlen', 10)
        mxwlen = kwargs.get('mxwlen', 40)

        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen

        token_buffer = tokens
        tokens_seq = tokens
        examples = dict()
        for i in range(mxlen):

            for k, vectorizer in self.vectorizers.items():
                vectorizer.mxlen = len(token_buffer)
                vec, length = vectorizer.run(token_buffer, self.vocabs[k])
                if k in examples:
                    examples[k] = np.append(examples[k], vec)
                else:
                    examples[k] = vec

                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key in examples:
                        examples[lengths_key] += length
                    else:
                        examples[lengths_key] = np.array(length)
            batch_dict = {k: v.reshape((1,) + v.shape) for k, v in examples.items()}
            softmax_tokens = self.model.predict(batch_dict)
            next_token = np.argmax(softmax_tokens, axis=-1)[-1]

            token_str = self.idx_to_token.get(next_token, '<PAD>')
            if token_str == '<EOS>':
                break
            if token_str != '<PAD>':
                tokens_seq += [token_str]
            token_buffer = [token_str]
        return tokens_seq


@exporter
class EncoderDecoderService(Service):

    @classmethod
    def task_name(cls):
        return 'seq2seq'

    @classmethod
    def signature_name(cls):
        return 'suggest_text'

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        super(EncoderDecoderService, self).__init__(None, None, model)
        self.src_vocabs = {}
        self.tgt_vocab = None
        for k, vocab in vocabs.items():
            if k == 'tgt':
                self.tgt_vocab = vocab
            else:
                self.src_vocabs[k] = vocab

        self.tgt_idx_to_token = revlut(self.tgt_vocab)
        self.src_vectorizers = {}
        self.tgt_vectorizer = None
        for k, vectorizer, in vectorizers.items():
            if k == 'tgt':
                self.tgt_vectorizer = vectorizer
            else:
                self.src_vectorizers[k] = vectorizer

    def src_vocab(self, vocab_type):
        return self.src_vocabs[vocab_type]

    def get_tgt_vocab(self):
        return self.tgt_vocab

    @classmethod
    def load(cls, bundle, **kwargs):
        kwargs['predict'] = kwargs.get('predict', True)
        kwargs['beam'] = kwargs.get('beam', 10)
        return super(EncoderDecoderService, cls).load(bundle, **kwargs)

    def set_vectorizer_lens(self, mxlen, mxwlen):
        for k, vectorizer in self.src_vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen

    def vectorize(self, tokens_seq):
        examples = defaultdict(list)
        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.src_vectorizers.items():
                vec, length = vectorizer.run(tokens, self.src_vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    examples[lengths_key] += [length]

        for k in self.src_vectorizers.keys():
            examples[k] = np.stack(examples[k])
            lengths_key = '{}_lengths'.format(k)
            if lengths_key in examples:
                examples[lengths_key] = np.array(examples[lengths_key])
        return examples

    def predict(self, tokens, K=1, **kwargs):
        tokens_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        examples = self.vectorize(tokens_seq)

        kwargs['beam'] = kwargs.get('beam', K)
        outcomes = self.model.predict(examples, **kwargs)

        results = []
        B = len(outcomes)
        for i in range(B):
            N = len(outcomes[i])
            n_best_result = []
            for n in range(min(K, N)):
                n_best = outcomes[i][n]
                out = lookup_sentence(self.tgt_idx_to_token, n_best).split()
                if K == 1:
                    results += [out]
                else:
                    n_best_result += [out]
            if K > 1:
                results.append(n_best_result)
        return results
