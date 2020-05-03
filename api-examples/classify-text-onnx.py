from __future__ import print_function

import baseline as bl
import argparse
import os
from baseline.utils import str2bool, read_json
import numpy as np
from collections import defaultdict
import onnxruntime as ort
from baseline.utils import unzip_files, find_model_basename, load_vectorizers, load_vocabs, put_addons_in_path


class ONNXClassifierService:

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
        outcomes_list = self.model.run(None, examples)[0]
        return self.format_output(outcomes_list)

    def format_output(self, predicted):
        results = []
        for outcomes in predicted:
            outcomes = list(zip(self.label_vocab, outcomes))
            results += [list(map(lambda x: (x[0], x[1].item()), sorted(outcomes, key=lambda tup: tup[1], reverse=True)))]
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
        labels = read_json(model_basename + '.labels')

        model = ort.InferenceSession(model_name)
        return cls(vocabs, vectorizers, model, labels)




parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='An ONNX bundle directory', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--conll', help='is file type conll?', type=str2bool, default=False)
parser.add_argument('--features', help='(optional) features in the format feature_name:index (column # in conll) or '
                                       'just feature names (assumed sequential)', default=[], nargs='+')
parser.add_argument('--device', help='device')
parser.add_argument('--batchsz', help='batch size when --text is a file', default=100, type=int)
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')
parser.add_argument('--export_mapping', help='mapping between features and the fields in the grpc/ REST '
                                                         'request, eg: token:word ner:ner. This should match with the '
                                                         '`exporter_field` definition in the mead config',
                    default=[], nargs='+')
parser.add_argument("--addon_path", type=str, default=os.path.expanduser('~/.bl-data/addons'),
                    help="Path or url of the dataset cache")
args = parser.parse_args()


put_addons_in_path(args.addon_path)
if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]
batched = [texts[i:i + args.batchsz] for i in range(0, len(texts), args.batchsz)]

m = ONNXClassifierService.load(args.model,
                               name=args.name, preproc=args.preproc,
                               device=args.device)

for texts in batched:
    for text, output in zip(texts, m.predict(texts)):
        print("{}, {}".format(" ".join(text), output[0][0]))

