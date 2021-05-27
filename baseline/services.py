import os
import pickle
import logging
from copy import deepcopy
from typing import Optional, List
from collections import defaultdict
import numpy as np
import baseline
from baseline.utils import (
    exporter,
    unzip_files,
    find_model_basename,
    find_files_with_prefix,
    import_user_module,
    read_json,
    is_sequence,
    Offsets,
    topk,
    to_numpy,
    revlut,
    load_vectorizers,
    load_vocabs,
    lookup_sentence,
    normalize_backend,
)
from baseline.model import load_model_for


logger = logging.getLogger('baseline')
__all__ = []
export = exporter(__all__)


class Service:

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

    def format_output(self, predicted, **kwargs):
        """Turn the results of self.model.predict into our output format

        :param predicted: The results from self.model.predict
        :returns: Formatted output, different for each task
        """

    def batch_input(self, tokens):
        """Turn the input into a consistent format.
        :param tokens: tokens in format List[str] or List[List[str]]
        :return: List[List[str]]
        """
        # If the input is List[str] wrap it in list to make a batch of size one.
        return (tokens,) if isinstance(tokens[0], str) else tokens

    def prepare_vectorizers(self, tokens_batch):
        """Batch the input tokens, and call reset and count method on each vectorizers to set up their mxlen.
           This method is mainly for reducing repeated code blocks.

        :param tokens_batch: input tokens in format or List[List[str]]
        """
        for vectorizer in self.vectorizers.values():
            vectorizer.reset()
            for tokens in tokens_batch:
                _ = vectorizer.count(tokens)

    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        for i, tokens in enumerate(tokens_batch):
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
        basehead = None

        if os.path.isdir(bundle):
            directory = bundle
        elif os.path.isfile(bundle):
            directory = unzip_files(bundle)
        else:
            directory = os.path.dirname(bundle)
            basehead = os.path.basename(bundle)
        model_basename = find_model_basename(directory, basehead)
        suffix = model_basename.split('-')[-1] + ".json"
        vocabs = load_vocabs(directory, suffix)

        be = normalize_backend(kwargs.get('backend', 'tf'))

        remote = kwargs.get("remote", None)
        name = kwargs.get("name", None)
        if remote:
            logging.debug("loading remote model")
            beam = int(kwargs.get('beam', 30))
            model, preproc = Service._create_remote_model(
                directory, be, remote, name, cls.task_name(), cls.signature_name(), beam,
                preproc=kwargs.get('preproc', 'client'),
                version=kwargs.get('version'),
                remote_type=kwargs.get('remote_type'),
            )
            vectorizers = load_vectorizers(directory)
            return cls(vocabs, vectorizers, model, preproc)

        # Currently nothing to do here
        # labels = read_json(os.path.join(directory, model_basename) + '.labels')

        import_user_module('baseline.{}.embeddings'.format(be))
        try:
            import_user_module('baseline.{}.{}'.format(be, cls.task_name()))
        except:
            pass
        model = load_model_for(cls.task_name(), model_basename, **kwargs)
        vectorizers = load_vectorizers(directory)
        return cls(vocabs, vectorizers, model, 'client')

    @staticmethod
    def _create_remote_model(directory, backend, remote, name, task_name, signature_name, beam, **kwargs):
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

        if backend not in {'tf', 'onnx'}:
            raise ValueError(f"Unsupported backend {backend} for remote Services")
        import_user_module('baseline.{}.remote'.format(backend))
        exp_type = kwargs.get('remote_type')
        if exp_type is None:
            exp_type = 'http' if remote.startswith('http') else 'grpc'
            exp_type = '{}-preproc'.format(exp_type) if preproc == 'server' else exp_type
            exp_type = f'{exp_type}-{task_name}'
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


@export
class ClassifierService(Service):
    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super().__init__(vocabs, vectorizers, model, preproc)
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

    @classmethod
    def load(cls, bundle, **kwargs):
        backend = kwargs.get('backend', 'tf')
        remote = kwargs.get('remote')
        if backend == 'onnx' and remote is None:
            return ONNXClassifierService.load(bundle, **kwargs)
        return super().load(bundle, **kwargs)

    def predict(self, tokens, preproc=None, raw=False, dense=False):
        """Take tokens and apply the internal vocab and vectorizers.  The tokens should be either a batch of text
        single utterance of type ``list``
        """
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `ClassifierService.predict` is deprecated.")
        if raw and not dense:
            logger.warning("Warning: `raw` parameter is deprecated pass `dense=True` to get back values as a single tensor")
            dense = True
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        if self.preproc == "client":
            examples = self.vectorize(tokens_batch)
        elif self.preproc == 'server':
            # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
            # vectorizers should not be available when preproc=server.
            featurized_examples = self.vectorize(tokens_batch)
            examples = {
                        'tokens': np.array([" ".join(x) for x in tokens_batch]),
                        self.model.lengths_key: featurized_examples[self.model.lengths_key]
            }

        outcomes_list = self.model.predict(examples, dense=dense)
        return self.format_output(outcomes_list, dense=dense)

    def format_output(self, predicted, dense=False, **kwargs):
        if dense:
            return predicted
        results = []
        for outcomes in predicted:
            if self.return_labels:
                results += [list(map(lambda x: (x[0], x[1].item()), sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
            else:
                # We have a list of tuples, one per class, sort these by score
                outcomes = sorted(outcomes, key=lambda tup: tup[1], reverse=True)

                results += [list(map(lambda x: (self.label_vocab[x[0].item()], x[1].item()),
                                     outcomes))]
        return results


@export
class ONNXClassifierService(ClassifierService):

    def __init__(self, vocabs=None, vectorizers=None, model=None, labels=None, lengths_key=None, **kwargs):
        self.label_vocab = labels
        self.lengths_key = lengths_key
        super().__init__(vocabs, vectorizers, model, **kwargs)
        self.return_labels = False
        self.input_names = set([x.name for x in model.get_inputs()])

    def get_labels(self):
        return self.labels

    def predict(self, tokens, **kwargs):
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # Hide the fact we can only do one at a time
        examples = [self.vectorize([tokens]) for tokens in tokens_batch]
        outcomes_list = np.concatenate([self.model.run(None, example)[0] for example in examples])
        return self.format_output(outcomes_list, dense=kwargs.get('dense', False))

    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        if self.lengths_key is None and 'lengths' in self.input_names:
            self.lengths_key = list(self.vectorizers.keys())[0]

        for i, tokens in enumerate(tokens_batch):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if self.lengths_key == k and length:
                    #examples[f'{self.lengths_key}_lengths'].append(length)
                    examples['lengths'].append(length)
        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
        if 'lengths' in examples:
            examples['lengths'] = np.stack(examples['lengths'])
        return examples

    def format_output(self, predicted, dense=False):
        if dense:
            return predicted
        results = []
        for outcomes in predicted:
            outcomes = list(zip(self.label_vocab, outcomes))
            results += [list(map(lambda x: (x[0], x[1].item()), sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
        return results

    @classmethod
    def load(cls, bundle, **kwargs):
        """Load a model from a bundle.

        This can be either a local model or a remote, exported model.

        :returns a Service implementation
        """
        import onnxruntime as ort

        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        # Try and unzip if its a zip file
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory)
        # model_basename = model_basename.replace(".pyt", "")
        model_name = f"{model_basename}.onnx"

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        # Currently nothing to do here
        labels = read_json(model_basename + '.labels')

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model, labels)

@export
class ONNXEmbeddingService(Service):

    def __init__(self, vocabs=None, vectorizers=None, model=None, lengths_key=None, **kwargs):
        super().__init__(vocabs, vectorizers, model,)
        self.lengths_key = lengths_key
        self.input_names = set([x.name for x in model.get_inputs()])

    def predict(self, tokens, **kwargs):
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # Hide the fact we can only do one at a time
        examples = [self.vectorize([tokens]) for tokens in tokens_batch]
        outcomes_list = np.concatenate([self.model.run(None, example)[0] for example in examples])
        return outcomes_list

    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        if self.lengths_key is None and 'lengths' in self.input_names:
            self.lengths_key = list(self.vectorizers.keys())[0]

        for i, tokens in enumerate(tokens_batch):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if self.lengths_key == k and length:
                    #examples[f'{self.lengths_key}_lengths'].append(length)
                    examples['lengths'].append(length)
        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
        if 'lengths' in examples:
            examples['lengths'] = np.stack(examples['lengths'])
        return examples

    @classmethod
    def load(cls, bundle, **kwargs):
        """Load a model from a bundle.

        This can be either a local model or a remote, exported model.

        :returns a Service implementation
        """
        import onnxruntime as ort

        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        # Try and unzip if its a zip file
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory)
        # model_basename = model_basename.replace(".pyt", "")
        model_name = f"{model_basename}.onnx"

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model)


@export
class EmbeddingsService(Service):
    @classmethod
    def task_name(cls):
        return 'servable-embeddings'

    @classmethod
    def signature_name(cls):
        return 'embed_text'

    def predict(self, tokens, preproc=None):
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `EmbeddingsService.predict` is deprecated.")
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        if self.preproc == 'client':
            examples = self.vectorize(tokens_batch)
        else:
            examples = {
                'tokens': np.array([" ".join(x) for x in tokens_batch]),
            }
        return self.format_output(self.model.predict(examples))

    def format_output(self, predicted):
        return predicted

    @classmethod
    def load(cls, bundle, **kwargs):
        import_user_module('hub:v1:addons:create_servable_embeddings')
        return super().load(bundle, **kwargs)


@export
class TaggerService(Service):

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super().__init__(vocabs, vectorizers, model, preproc)
        if hasattr(self.model, 'return_labels'):
            self.return_labels = self.model.return_labels
        else:
            self.return_labels = False  # keeping the default tagger behavior
        if not self.return_labels:
            self.label_vocab = revlut(self.get_labels())
        self.rev_vocab = {k: revlut(v) for k, v in self.vocabs.items()}

    @classmethod
    def task_name(cls):
        return 'tagger'

    @classmethod
    def signature_name(cls):
        return 'tag_text'

    @classmethod
    def load(cls, bundle, **kwargs):
        backend = kwargs.get('backend', 'tf')
        remote = kwargs.get('remote')
        if backend == 'onnx' and remote is None:
            return ONNXTaggerService.load(bundle, **kwargs)
        return super().load(bundle, **kwargs)

    def batch_input(self, tokens):
        """Convert the input into a consistent format.

        :return: List[List[dict[str] -> str]]
        """
        # Input is a list of strings. (assume strings are tokens)
        if isinstance(tokens[0], str):
            tokens_batch = []
            for t in tokens:
                tokens_batch.append({'text': t})
            tokens_batch = [tokens_batch]
        else:
            # Better be a sequence, but it could be pre-batched, [[],[]]
            # But what kind of object is at the first one then?
            if is_sequence(tokens[0]):
                tokens_batch = []
                # Then what we have is [['The', 'dog',...], ['I', 'cannot']]
                # [[{'text': 'The', 'pos': 'DT'}, ...

                # For each of the utterances, we need to make a dictionary
                if isinstance(tokens[0][0], str):
                    for utt in tokens:
                        utt_dict_seq = []
                        for t in utt:
                            utt_dict_seq += [dict({'text': t})]
                        tokens_batch += [utt_dict_seq]
                # Its already in List[List[dict]] form, do nothing
                elif isinstance(tokens[0][0], dict):
                    tokens_batch = tokens
            # If its a dict, we just wrap it up
            elif isinstance(tokens[0], dict):
                tokens_batch = [tokens]
            else:
                raise Exception('Unknown input format')

        if len(tokens_batch) == 0:
            return []
        return tokens_batch

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
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
        # vectorizers should not be available when preproc=server.
        examples = self.vectorize(tokens_batch)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                   for y in x]) for x in tokens_batch])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        outcomes = self.model.predict(examples)
        return self.format_output(outcomes, tokens_batch=tokens_batch, label_field=label_field, vectorized_examples=examples)

    def format_output(self, predicted, tokens_batch=None, label_field='label', vectorized_examples=None, **kwargs):
        """This code got very messy dealing with BPE/WP outputs."""
        assert tokens_batch is not None
        assert vectorized_examples is not None
        outputs = []
        # Pick a random non-lengths key from the vectorized input, if one input was BPE'd they all had to be to stay aligned
        key = [k for k in vectorized_examples if not k.endswith("_lengths")][0]
        # For each item in a batch
        for i, outcome in enumerate(predicted):
            output = []
            # Extract the vectorized example for this batch element and the key we are choosing
            vectorized_example = vectorized_examples[key][i]
            # Convert back into strings, these will now be broken into subwords
            tokenized_text = [self.rev_vocab[key][t] for t in vectorized_example]
            new_outcome = [
                outcome[j] if self.return_labels else self.label_vocab[outcome[j].item()]
                for j in self.vectorizers[key].valid_label_indices(tokenized_text)
            ]
            # Loop through the (now aligned) og tokens and the labels
            for token, label in zip(tokens_batch[i], new_outcome):
                new_token = deepcopy(token)
                # Our labels now have <PAD> in their vocab, if we see one just hide it with an "O"
                label = "O" if label == Offsets.VALUES[Offsets.PAD] else label
                new_token[label_field] = label
                output.append(new_token)
            outputs.append(output)
        return outputs


class ONNXTaggerService(TaggerService):

    def __init__(self, vocabs=None, vectorizers=None, model=None, labels=None, lengths_key=None, **kwargs):
        self.labels = labels
        self.lengths_key = lengths_key
        super().__init__(vocabs, vectorizers, model)
        self.input_names = set([x.name for x in model.get_inputs()])

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.labels

    def predict(self, tokens, **kwargs):
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # Process each example in the batch by itself to hide the one at a time nature
        examples = [self.vectorize([tokens]) for tokens in tokens_batch]
        outcomes_list = np.concatenate([self.model.run(None, example)[0] for example in examples], axis=0)
        return self.format_output(outcomes_list, tokens_batch, label_field=kwargs.get('label', 'label'), vectorized_examples=examples)

    def format_output(self, *args, **kwargs):
        """Because the ONNX service is hiding it's one at a time nature it is a List[Dict[str]] instead of a Dict[str, List]
           So flip it so it can be processed by the normal format_output.
        """
        vectorized_examples = kwargs['vectorized_examples']
        new_vec = defaultdict(list)
        for key in vectorized_examples[0]:
            for ve in vectorized_examples:
                new_vec[key].append(ve[key][0])
        kwargs['vectorized_examples'] = new_vec
        return super().format_output(*args, **kwargs)

    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        if self.lengths_key is None and 'lengths' in self.input_names:
            self.lengths_key = list(self.vectorizers.keys())[0]

        for i, tokens in enumerate(tokens_batch):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if self.lengths_key == k and length:
                    examples['lengths'].append(length)
        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
        if 'lengths' in examples:
            examples['lengths'] = np.stack(examples['lengths'])
        return examples

    @classmethod
    def load(cls, bundle, **kwargs):
        """Load a model from a bundle.

        This can be either a local model or a remote, exported model.

        :returns a Service implementation
        """
        import onnxruntime as ort
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory)
        model_name = f"{model_basename}.onnx"

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        # Currently nothing to do here
        labels = read_json(model_basename + '.labels')

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model, labels)


@export
class DependencyParserService(Service):

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super().__init__(vocabs, vectorizers, model, preproc)
        # The model always returns indices (no need for `return_labels`)
        self.label_vocab = revlut(self.get_labels())
        self.rev_vocab = {k: revlut(v) for k, v in self.vocabs.items()}

    @classmethod
    def task_name(cls):
        return 'deps'

    @classmethod
    def signature_name(cls):
        return 'deps_text'

    @classmethod
    def load(cls, bundle, **kwargs):
        backend = kwargs.get('backend', 'tf')
        if backend == 'onnx':
            return ONNXDependencyParserService.load(bundle, **kwargs)
        return super().load(bundle, **kwargs)

    def batch_input(self, tokens):
        """Convert the input into a consistent format.

        :return: List[List[dict[str] -> str]]
        """
        # Input is a list of strings. (assume strings are tokens)
        if isinstance(tokens[0], str):
            tokens_batch = []
            for t in tokens:
                tokens_batch.append({'text': t})
            tokens_batch = [tokens_batch]
        else:
            # Better be a sequence, but it could be pre-batched, [[],[]]
            # But what kind of object is at the first one then?
            if is_sequence(tokens[0]):
                tokens_batch = []
                # Then what we have is [['The', 'dog',...], ['I', 'cannot']]
                # [[{'text': 'The', 'pos': 'DT'}, ...

                # For each of the utterances, we need to make a dictionary
                if isinstance(tokens[0][0], str):
                    for utt in tokens:
                        utt_dict_seq = []
                        for t in utt:
                            utt_dict_seq += [dict({'text': t})]
                        tokens_batch += [utt_dict_seq]
                # Its already in List[List[dict]] form, do nothing
                elif isinstance(tokens[0][0], dict):
                    tokens_batch = tokens
            # If its a dict, we just wrap it up
            elif isinstance(tokens[0], dict):
                tokens_batch = [tokens]
            else:
                raise Exception('Unknown input format')

        if len(tokens_batch) == 0:
            return []
        return tokens_batch

    def predict(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the parser.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        examples = self.vectorize(tokens_batch)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                            for y in x]) for x in tokens_batch])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        outcomes = self.model.predict(examples)
        return self.format_output(outcomes, tokens_batch=tokens_batch, label_field=label_field, vectorized_examples=examples)

    def format_output(self, predicted, tokens_batch=None, label_field='label', arc_field='head', vectorized_examples=None, **kwargs):
        assert tokens_batch is not None
        assert vectorized_examples is not None
        outputs = []
        # For each item in a batch
        arcs, labels = predicted
        for i, (arc_list, label_list) in enumerate(zip(arcs, labels)):
            output = []
            for token, arc, label in zip(tokens_batch[i], arc_list[1:], label_list[1:]):
                new_token = deepcopy(token)
                new_token[label_field] = self.label_vocab[label.item()]
                new_token[arc_field] = arc.item()
                output.append(new_token)
            outputs.append(output)
        return outputs


class ONNXDependencyParserService(DependencyParserService):

    def __init__(self, vocabs=None, vectorizers=None, model=None, labels=None, lengths_key=None, **kwargs):
        self.labels = labels
        self.lengths_key = lengths_key
        super().__init__(vocabs, vectorizers, model)

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.labels

    def predict(self, tokens, **kwargs):
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # Process each example in the batch by itself to hide the one at a time nature
        examples = [self.vectorize([tokens]) for tokens in tokens_batch]
        arcs_batch = []
        labels_batch = []
        for example in examples:
            arcs_logits, labels_logits = self.model.run(None, example)
            arcs_logits = arcs_logits.squeeze(0)
            labels_logits = labels_logits.squeeze(0)
            arcs = np.argmax(arcs_logits, -1)
            labels = np.argmax(labels_logits[np.arange(len(arcs)), arcs], -1)
            arcs_batch.append(arcs)
            labels_batch.append(labels)
        return self.format_output((arcs_batch, labels_batch), tokens_batch, label_field=kwargs.get('label', 'label'), vectorized_examples=examples)

    def format_output(self, *args, **kwargs):
        """Because the ONNX service is hiding it's one at a time nature it is a List[Dict[str]] instead of a Dict[str, List]
           So flip it so it can be processed by the normal format_output.
        """
        vectorized_examples = kwargs['vectorized_examples']
        new_vec = defaultdict(list)
        for key in vectorized_examples[0]:
            for ve in vectorized_examples:
                new_vec[key].append(ve[key][0])
        kwargs['vectorized_examples'] = new_vec
        return super().format_output(*args, **kwargs)

    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        if self.lengths_key is None:
            self.lengths_key = list(self.vectorizers.keys())[0]

        for i, tokens in enumerate(tokens_batch):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if self.lengths_key == k and length:
                    examples['lengths'].append(length)
        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
        if 'lengths' in examples:
            examples['lengths'] = np.stack(examples['lengths'])
        return examples

    @classmethod
    def load(cls, bundle, **kwargs):
        """Load a model from a bundle.

        This can be either a local model or a remote, exported model.

        :returns a Service implementation
        """
        import onnxruntime as ort
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory)
        model_name = f"{model_basename}.onnx"

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        # Currently nothing to do here
        labels = read_json(model_basename + '.labels')

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model, labels)


@export
class LanguageModelService(Service):

    @classmethod
    def task_name(cls):
        return "lm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_token = revlut(self.vocabs[self.model.tgt_key])

    @classmethod
    def load(cls, bundle, **kwargs):
        kwargs['batchsz'] = 1
        return super().load(bundle, **kwargs)

    def conditional(self, context, target: Optional[List[str]] = None, limit: Optional[int] = None, raw: bool = False, **kwargs):
        """Get the conditional probabilities of the next tokens.

        :param context: The tokens
        :param target: A list of values that you want the conditional prob of P(target | context)
        :param limit: The number of (next word, score) pairs to return
        :param raw: Should you just return the raw softmax values? This will override the limit argument

        :returns: The conditional probs of a specific target word if `target` is defined, the top limit softmax scores
            for the next step, or the raw softmax numpy array if `raw` is set
        """
        if kwargs.get('preproc', None) is not None:
            logger.warning("Warning: Passing `preproc` to `LanguageModelService.predict` is deprecated.")
        tokens_batch = self.batch_input(context)
        self.prepare_vectorizers(tokens_batch)
        batch_dict = self.vectorize(tokens_batch)
        next_softmax = self.model.predict(batch_dict)[:, -1, :]
        next_softmax = to_numpy(next_softmax)
        if target is not None:
            target_batch = [[t] for t in target]
            self.prepare_vectorizers(target_batch)
            target_batch = self.vectorize(target_batch)
            target = target_batch[self.model.tgt_key]
            return np.array([v.item() for v in next_softmax[np.arange(next_softmax.shape[0]), target]])
        if raw:
            return next_softmax
        limit = next_softmax.shape[-1] if limit is None else limit
        scores = [topk(limit, soft) for soft in next_softmax]
        return [{self.idx_to_token[k]: v for k, v in score.items()} for score in scores]

    @staticmethod
    def pad_eos(tokens_batch):
        """Add <EOS> tokens to both the beginning of each item in the batch.

        Note:
            When training the language models we have and <EOS> token between each sentence we use
            that here to represent these tokens ending where they do. Because each sentence end with
            <EOS> the next sentence always starts with the <EOS> so we add that here too.
        """
        return [[Offsets.VALUES[Offsets.EOS]] + t + [Offsets.VALUES[Offsets.EOS]] for t in tokens_batch]

    def joint(self, tokens, **kwargs):
        """Score tokens with a language model.

        Note:
            This is not quite the correct joint probability I think, the joint prob of P(wn, wn-1, ... w1)
            is P(w_1) * \pi_2^n P(w_i| w_i-1, ... w_1) but here we have the P(w_1 | <EOS>) which is the
            prob of w_1 staring a string, not quite the same as the unconditional probability.

        :tokens: A sequence of tokens

        returns: The score based on the probability type
        """
        if kwargs.get('preproc', None) is not None:
            logger.warning("Warning: Passing `preproc` to `LanguageModelService.predict` is deprecated.")
        tokens_batch = self.batch_input(tokens)
        tokens_batch = self.pad_eos(tokens_batch)
        self.prepare_vectorizers(tokens_batch)
        batch_dict = self.vectorize(tokens_batch)
        softmax_tokens = self.model.predict(batch_dict)

        # Get the targets, the first is a <EOS> seed so we skip that
        values = batch_dict[self.model.tgt_key][:, 1:]

        # Numpy doesn't have a gather so we can't do this very efficiently
        # scores = torch.gather(softmax_tokens, -1, values.unsqueeze(-1))
        scores = []
        # The last softmax is what comes after the <EOS> so we don't grab from there
        for soft, value in zip(softmax_tokens[:, :-1], values):
            tokens = []
            for tok, val in zip(soft, value):
                tokens.append(to_numpy(tok)[val].item())
            scores.append(tokens)
        scores = np.array(scores)
        return np.exp(np.sum(np.log(scores), axis=1))

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
            next_token = np.argmax(softmax_tokens[:, -1, :], axis=-1)[0]
            token_str = self.idx_to_token.get(next_token, '<PAD>')
            if token_str == '<EOS>':
                break
            if token_str != '<PAD>':
                tokens_seq += [token_str]
            token_buffer = [token_str]
        return tokens_seq


@export
class EncoderDecoderService(Service):

    @classmethod
    def task_name(cls):
        return 'seq2seq'

    @classmethod
    def signature_name(cls):
        return 'suggest_text'

    def __init__(self, vocabs=None, vectorizers=None, model=None, preproc='client'):
        super().__init__(None, None, model, preproc)
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
        return super().load(bundle, **kwargs)

    def vectorize(self, tokens_batch):
        examples = defaultdict(list)
        for i, tokens in enumerate(tokens_batch):
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
        tokens_batch = self.batch_input(tokens)
        for vectorizer in self.src_vectorizers.values():
            vectorizer.reset()
            for tokens in tokens_batch:
                _ = vectorizer.count(tokens)
        examples = self.vectorize(tokens_batch)

        kwargs['beam'] = int(kwargs.get('beam', K))
        outcomes, scores = self.model.predict(examples, **kwargs)
        return self.format_output(outcomes, K=K)

    def format_output(self, predicted, K=1, **kwargs):
        results = []
        B = len(predicted)
        for i in range(B):
            N = len(predicted[i])
            n_best_result = []
            for n in range(min(K, N)):
                n_best = predicted[i][n]
                out = lookup_sentence(self.tgt_idx_to_token, n_best).split()
                if K == 1:
                    results += [out]
                else:
                    n_best_result += [out]
            if K > 1:
                results.append(n_best_result)
        return results
