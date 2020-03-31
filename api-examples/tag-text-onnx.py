from __future__ import print_function
from collections import defaultdict
import baseline as bl
import numpy as np
import argparse
import os
from baseline.utils import str2bool, read_conll, read_json, revlut
import onnxruntime as ort
from baseline.utils import unzip_files, find_model_basename, load_vectorizers, load_vocabs


class ONNXTaggerService(object):

    def __init__(self, vocabs=None, vectorizers=None, model=None, labels=None, lengths_key=None):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs
        self.label_vocab = labels
        self.lengths_key = lengths_key

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.model.get_labels()

    def predict(self, tokens, **kwargs):
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        examples = self.vectorize(tokens_batch)
        outcomes_list = self.model.run(None, examples)
        outcomes_list = outcomes_list[0]
        return self.format_output(tokens, outcomes_list)

    def format_output(self, tokens, predicted):
        results = []
        for inputs, outcomes in zip(tokens, predicted):
            outcomes = [{'text': x, 'label': self.label_vocab[y]} for x, y in zip(inputs, outcomes)]
            results.append(outcomes)
        return results

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
        if self.lengths_key is None:
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
        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        model_basename = find_model_basename(directory).replace(".pyt", "")
        model_name = f"{model_basename}.onnx"

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        # Currently nothing to do here
        labels = revlut(read_json(model_basename + '.labels'))

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model, labels)



parser = argparse.ArgumentParser(description='Tag text with a model')
parser.add_argument('--model', help='A tagger model with extended features', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--conll', help='is file type conll?', type=str2bool, default=False)
parser.add_argument('--features', help='(optional) features in the format feature_name:index (column # in conll) or '
                                       'just feature names (assumed sequential)', default=[], nargs='+')
parser.add_argument('--device', help='device')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')
parser.add_argument('--export_mapping', help='mapping between features and the fields in the grpc/ REST '
                                                         'request, eg: token:word ner:ner. This should match with the '
                                                         '`exporter_field` definition in the mead config',
                    default=[], nargs='+')
args = parser.parse_args()


def create_export_mapping(feature_map_strings):
    feature_map_strings = [x.strip() for x in feature_map_strings if x.strip()]
    if not feature_map_strings:
        return {}
    else:
        return {x[0]: x[1] for x in [y.split(':') for y in feature_map_strings]}


def feature_index_mapping(features):
    if not features:
        return {}
    elif ':' in features[0]:
        return {feature.split(':')[0]: int(feature.split(':')[1]) for feature in features}
    else:
        return {feature: index for index, feature in enumerate(features)}


if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    if args.conll:
        feature_indices = feature_index_mapping(args.features)
        for sentence in read_conll(args.text):
            if feature_indices:
                texts.append([{k: line[v] for k, v in feature_indices.items()} for line in sentence])
            else:
                texts.append([line[0] for line in sentence])
    else:
        with open(args.text, 'r') as f:
            for line in f:
                text = line.strip().split()
                texts += [text]
else:
    texts = [args.text.split()]

m = ONNXTaggerService.load(args.model)
for sen in m.predict(texts, export_mapping=create_export_mapping(args.export_mapping)):
    for word_tag in sen:
        print("{} {}".format(word_tag['text'], word_tag['label']))
    print()
