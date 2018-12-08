import os
import pickle
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
    load_vocabs
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

        be = kwargs.get('backend', 'tf')

        remote = kwargs.get("remote", None)
        name = kwargs.get("name", None)
        if remote:
            beam = kwargs.get('beam', 10)
            model = Service._create_remote_model(directory, be, remote, name, cls.signature_name(), beam)
            return cls(vocabs, vectorizers, model)

        # Currently nothing to do here
        # labels = read_json(os.path.join(directory, model_basename) + '.labels')

        import_user_module('baseline.{}.embeddings'.format(be))
        import_user_module('baseline.{}.{}'.format(be, cls.task_name()))
        model = load_model_for(cls.task_name(), model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

    @staticmethod
    def _create_remote_model(directory, backend, remote, name, signature_name, beam):
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
            RemoteModel = remote_models.RemoteModelTensorFlowREST if remote.startswith('http') else remote_models.RemoteModelTensorFlowGRPC
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

    def predict(self, tokens):
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
                examples[k].append(vec)
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key].append(length)

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])

        outcomes_list = self.model.predict(examples)

        results = []
        for outcomes in outcomes_list:
            results += [sorted(outcomes, key=lambda tup: tup[1], reverse=True)]
        return results


@exporter
class TaggerService(Service):

    @classmethod
    def task_name(cls):
        return 'tagger'

    @classmethod
    def signature_name(cls):
        return 'tag_text'

    def predict(self, tokens, **kwargs):
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
                    examples[lengths_key] += [np.int32(length)]

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
            lengths_key = '{}_lengths'.format(k)
            examples[lengths_key] = np.stack(examples[lengths_key])

        outcomes = self.model.predict(examples)
        outputs = []
        for i, outcome in enumerate(outcomes):
            output = []
            for j, token in enumerate(tokens_seq[i]):
                new_token = dict()
                new_token.update(token)
                new_token[label_field] = label_vocab[np.int32(outcome[j])]
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

    def predict(self, tokens, **kwargs):

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

        for k, vectorizer in self.src_vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.src_vectorizers.items():
                vec, length = vectorizer.run(tokens, self.src_vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in self.src_vectorizers.keys():
            examples[k] = np.stack(examples[k])

        outcomes = self.model.predict(examples)

        results = []
        ##B = outcomes.shape[0]
        for i in range(len(outcomes)):
            best = outcomes[i][0]
            out = []
            for j in range(len(best)):
                word = self.tgt_idx_to_token.get(best[j], '<PAD>')
                if word != '<PAD>' and word != '<EOS>':
                    out += [word]
            results += [out]
        return results
