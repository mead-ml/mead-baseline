import six

import os
import pickle
from collections import defaultdict
import numpy as np
import baseline
import logging
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
logger = logging.getLogger('baseline')


__all__ = []
exporter = export(__all__)


class Service(object):

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs
        self.preproc = preproc

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

    def predict(self, tokens, **kwargs):
        pass

    def batch_input(self, tokens):
        """Turn the input into a consistent format.
        :param tokens: tokens in format List[str] or List[List[str]]
        :return: List[List[str]]
        """
        mxlen = 0
        mxwlen = 0
        vmxlen, vmxwlen = self.get_vectorizer_lens()
        # If the input is List[str] wrap it in list to make a batch of size one.
        tokens_seq = (tokens,) if isinstance(tokens[0], six.string_types) else tokens
        # Get sentence and word lengths from the batch
        for tokens in tokens_seq:
            if vmxlen != -1:
                tokens = tokens[:vmxlen]
            mxlen = max(mxlen, len(tokens))
            for index in range(len(tokens)):
                if vmxwlen != -1:
                    tokens[index] = tokens[index][:vmxwlen]
                mxwlen = max(mxwlen, len(tokens[index]))
        return tokens_seq, mxlen, mxwlen

    def set_vectorizer_lens(self, mxlen, mxwlen):
        """Set the max lengths on the vectorizers if unset.

        :param mxlen: `int`: The max length of an example
        :param mxwlen: `int`: The max length of a word in the batch
        """
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen'):
                if self.preproc == 'client':
                    vectorizer.mxlen = mxlen
                elif self.preproc == 'server' and vectorizer.mxlen == -1:  # keep the original behavior
                    vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen'):
                if self.preproc == 'client':
                    vectorizer.mxwlen = mxwlen
                elif self.preproc == 'server' and vectorizer.mxwlen == -1:  # keep the original behavior
                    vectorizer.mxwlen = mxwlen

    def get_vectorizer_lens(self):
        """get the max len from the vectorizers if defined
        """
        mxlen = -1
        mxwlen = -1
        if self.preproc == 'server' and self.vectorizers is not None:
            for k, vectorizer in self.vectorizers.items():
                if hasattr(vectorizer, 'mxlen'):
                    mxlen = vectorizer.mxlen
                if hasattr(vectorizer, 'mxwlen'):
                    mxwlen = vectorizer.mxwlen
        return mxlen, mxwlen

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
            logging.debug("loading remote model")
            beam = int(kwargs.get('beam', 30))
            model, preproc = Service._create_remote_model(
                directory, be, remote, name, cls.signature_name(), beam,
                preproc=kwargs.get('preproc', 'client'),
                version=kwargs.get('version')
            )
            return cls(vocabs, vectorizers, model, preproc)

        # Currently nothing to do here
        # labels = read_json(os.path.join(directory, model_basename) + '.labels')

        import_user_module('baseline.{}.embeddings'.format(be))
        try:
            import_user_module('baseline.{}.{}'.format(be, cls.task_name()))
        except:
            pass
        model = load_model_for(cls.task_name(), model_basename, **kwargs)
        return cls(vocabs, vectorizers, model, 'client')

    @staticmethod
    def _create_remote_model(directory, backend, remote, name, signature_name, beam, **kwargs):
        """Reads the necessary information from the remote bundle to instatiate
        a client for a remote model.

        :directory the location of the exported model bundle
        :remote a url endpoint to hit
        :name the model name, as defined in tf-serving's model.config
        :signature_name  the signature to use.
        :beam used for s2s and found in the kwargs. We default this and pass it in.

        :returns a RemoteModel
        """
        from baseline.remote import create_remote
        assets = read_json(os.path.join(directory, 'model.assets'))
        model_name = assets['metadata']['exported_model']
        preproc = assets['metadata'].get('preproc', kwargs.get('preproc', 'client'))
        labels = read_json(os.path.join(directory, model_name) + '.labels')
        lengths_key = assets.get('lengths_key', None)
        inputs = assets.get('inputs', [])
        return_labels = bool(assets['metadata']['return_labels'])
        version = kwargs.get('version')

        if backend not in {'tf', 'pytorch'}:
            raise ValueError("only Tensorflow and Pytorch are currently supported for remote Services")
        import_user_module('baseline.{}.remote'.format(backend))
        exp_type = 'http' if remote.startswith('http') else 'grpc'
        exp_type = '{}-preproc'.format(exp_type) if preproc == 'server' else exp_type
        model = create_remote(
            exp_type,
            remote=remote, name=name,
            signature=signature_name,
            labels=labels,
            lengths_key=lengths_key,
            inputs=inputs,
            beam=beam,
            return_labels=return_labels,
            version=version,
        )
        return model, preproc


@exporter
class ClassifierService(Service):
    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super(ClassifierService, self).__init__(vocabs, vectorizers, model, preproc)
        if hasattr(self.model, 'return_labels'):
            self.return_labels = self.model.return_labels
        else:
            self.return_labels = True  # keeping the default classifier behavior
        if not self.return_labels:
            self.label_vocab = {index: label for index, label in enumerate(self.get_labels())}

    @classmethod
    def task_name(cls):
        return 'classify'

    @classmethod
    def signature_name(cls):
        return 'predict_text'

    def predict(self, tokens, preproc=None):
        """Take tokens and apply the internal vocab and vectorizers.  The tokens should be either a batch of text
        single utterance of type ``list``
        """
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `ClassifierService.predict` is deprecated.")
        tokens_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        if self.preproc == "client":
            examples = self.vectorize(tokens_seq)
        elif self.preproc == 'server':
            # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
            # vectorizers should not be available when preproc=server.
            featurized_examples = self.vectorize(tokens_seq)
            examples = {
                        'tokens': np.array([" ".join(x) for x in tokens_seq]),
                        self.model.lengths_key: featurized_examples[self.model.lengths_key]
            }

        outcomes_list = self.model.predict(examples)
        results = []
        for outcomes in outcomes_list:
            if self.return_labels:
                results += [list(map(lambda x: (x[0], x[1].item()), sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
            else:
                results += [list(map(lambda x: (self.label_vocab[x[0].item()], x[1].item()),
                                     sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
        return results

@exporter
class EmbeddingsService(Service):
    @classmethod
    def task_name(cls):
        return 'servable_embeddings'

    @classmethod
    def signature_name(cls):
        return 'embed_text'

    def predict(self, tokens, preproc=None):
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `EmbeddingsService.predict` is deprecated.")
        tokens_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        if self.preproc == 'client':
            examples = self.vectorize(tokens_seq)
        else:
            examples = {
                'tokens': np.array([" ".join(x) for x in tokens_seq]),
            }
        return self.model.predict(examples)

    @classmethod
    def load(cls, bundle, **kwargs):
        import_user_module('create_servable_embeddings')
        return super(EmbeddingsService, cls).load(bundle, **kwargs)


@exporter
class TaggerService(Service):

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super(TaggerService, self).__init__(vocabs, vectorizers, model, preproc)
        if hasattr(self.model, 'return_labels'):
            self.return_labels = self.model.return_labels
        else:
            self.return_labels = False  # keeping the default tagger behavior
        if not self.return_labels:
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
        vmxlen, vmxwlen = self.get_vectorizer_lens()
        # Input is a list of strings. (assume strings are tokens)
        if isinstance(tokens[0], six.string_types):
            if vmxlen != -1:
                tokens = tokens[:vmxlen]
            mxlen = len(tokens)
            tokens_seq = []
            for t in tokens:
                mxwlen = max(mxwlen, len(t))
                if vmxwlen != -1:
                    t = t[:vmxwlen]
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
                        if vmxlen != -1:
                            utt = utt[:vmxlen]
                        mxlen = max(mxlen, len(utt))
                        for t in utt:
                            if vmxwlen != -1:
                                t = t[:vmxwlen]
                            mxwlen = max(mxwlen, len(t))
                            utt_dict_seq += [dict({'text': t})]
                        tokens_seq += [utt_dict_seq]
                # Its already in List[List[dict]] form so just iterate to get mxlen and mxwlen
                elif isinstance(tokens[0][0], dict):
                    for utt_dict_seq in tokens:
                        if vmxlen != -1:
                            utt_dict_seq = utt_dict_seq[:vmxlen]
                        mxlen = max(mxlen, len(utt_dict_seq))
                        for token_dict in utt_dict_seq:
                            text = token_dict['text']
                            if vmxwlen != -1:
                                text = text[:vmxwlen]
                                token_dict['text'] = text
                            mxwlen = max(mxwlen, len(text))
                        tokens_seq += [utt_dict_seq]
            # If its a dict, we just wrap it up
            elif isinstance(tokens[0], dict):
                if vmxlen != -1:
                    tokens = tokens[:vmxlen]
                mxlen = len(tokens)
                for t in tokens:
                    text = t['text']
                    if vmxwlen != -1:
                        text = text[:vmxwlen]
                        t['text'] = text
                    mxwlen = max(mxwlen, len(text))
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
        preproc = kwargs.get('preproc', None)
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `TaggerService.predict` is deprecated.")
        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_seq, mxlen, mxwlen = self.batch_input(tokens)
        self.set_vectorizer_lens(mxlen, mxwlen)
        # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
        # vectorizers should not be available when preproc=server.
        examples = self.vectorize(tokens_seq)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                   for y in x]) for x in tokens_seq])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        outcomes = self.model.predict(examples)

        outputs = []
        for i, outcome in enumerate(outcomes):
            output = []
            for j, token in enumerate(tokens_seq[i]):
                new_token = dict()
                new_token.update(token)
                if self.return_labels:
                    new_token[label_field] = outcome[j]
                else:
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

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super(EncoderDecoderService, self).__init__(None, None, model, preproc)
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
        kwargs['beam'] = int(kwargs.get('beam', 30))
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

        kwargs['beam'] = int(kwargs.get('beam', K))
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
