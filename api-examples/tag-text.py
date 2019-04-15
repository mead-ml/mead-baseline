import baseline as bl
import argparse
import os
from baseline.utils import str2bool


parser = argparse.ArgumentParser(description='Tag text with a model')
parser.add_argument('--model', help='A tagger model with extended features', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--conll', help='is file type conll?', type=str2bool, default=False)
parser.add_argument('--features', help='(optional) features in the format feature_name:index (column # in conll) or '
                                       'just feature names (assumed sequential)', default=[], nargs='+')
parser.add_argument('--backend', help='backend', default='tf')
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
        sentence = []
        with open(args.text, 'r') as f:
            for line in f:
                if line.strip():
                    text = line.strip().split()
                    if feature_indices:
                        text = {feature: text[feature_indices[feature]] for feature in feature_indices}
                    else:
                        text = text[0]
                    sentence.append(text)
                else:
                    texts.append(sentence)
                    sentence = []
            if sentence:
                texts.append(sentence)
    else:
        with open(args.text, 'r') as f:
            for line in f:
                text = line.strip().split()
                texts += [text]
else:
    texts = [args.text.split()]

m = bl.TaggerService.load(args.model, backend=args.backend, remote=args.remote,
                          name=args.name, preproc=args.preproc, device=args.device)
for sen in m.predict(texts, export_mapping=create_export_mapping(args.export_mapping)):
    for word_tag in sen:
        print("{} {}".format(word_tag['text'], word_tag['label']))
    print("\n")
