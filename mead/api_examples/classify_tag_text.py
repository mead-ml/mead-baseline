from __future__ import print_function

import baseline as bl
import argparse
import os
from eight_mile.utils import str2bool, read_conll

def main():
    parser = argparse.ArgumentParser(description='Jointly Classify and Tag text with a model')
    parser.add_argument('--model', help='A tagger model with extended features', required=True, type=str)
    parser.add_argument('--text', help='raw value', type=str)
    parser.add_argument('--conll', help='is file type conll?', type=str2bool, default=False)
    parser.add_argument('--ignore_first_token', help='if the input file is in conll format, and the first token'
                                                     ' in the sentence is the class label, ignore it',
                                                     type=str2bool, default=False)
    parser.add_argument('--features', help='(optional) features in the format feature_name:index (column # in conll) or '
                                           'just feature names (assumed sequential)', default=[], nargs='+')
    parser.add_argument('--backend', help='backend', default='pytorch')
    parser.add_argument('--device', help='device')
    parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
    parser.add_argument('--name', help='(optional) signature name', type=str)
    parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')
    parser.add_argument('--export_mapping', help='mapping between features and the fields in the grpc/ REST '
                                                             'request, eg: token:word ner:ner. This should match with the '
                                                             '`exporter_field` definition in the mead config',
                        default=[], nargs='+')
    parser.add_argument('--modules', default=[], nargs="+")
    parser.add_argument('--out_fmt', nargs="+")
    parser.add_argument('--batchsz', default=8, help="How many examples to run through the model at once", type=int)

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
                if args.ignore_first_token:
                    sentence.pop(0)
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

    m = bl.JointTaggerService.load(args.model, backend=args.backend, remote=args.remote,
                              name=args.name, preproc=args.preproc, device=args.device)

    batched = [texts[i:i+args.batchsz] for i in range(0, len(texts), args.batchsz)]

    if args.out_fmt:
        out_fmt = lambda w: ' '.join(w[f] for f in args.out_fmt)
    else:
        out_fmt = lambda w: f"{w['text']} {w['label']}"
    for texts in batched:
        for class_prediction, tags_prediction in m.predict(texts, export_mapping=create_export_mapping(args.export_mapping)):
            print(f"Class Label:{class_prediction}")
            for word_tag in tags_prediction:

                print(out_fmt(word_tag))
            print()


if __name__ == '__main__':
    main()
