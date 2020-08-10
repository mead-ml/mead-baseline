import os
import re
from collections import Counter
import pandas as pd
import baseline
from baseline.reader import TSVSeqLabelReader, register_reader


@register_reader(task='classify', name='pandas')
class PandasReader(TSVSeqLabelReader):
    def __init__(self, vectorizers, trim=False, **kwargs):
        super().__init__(vectorizers, trim, **kwargs)
        self.label = kwargs.get('label', 'label')
        self.text = kwargs.get('text', 'text')
        self.sep = kwargs.get('sep', ',')
        self.header = kwargs.get('header', 'infer')

    def build_vocab(self, files, **kwargs):
        label_idx = len(self.label2index)
        if isinstance(files, str):
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]

        vocab = dict()
        for k in self.vectorizers.keys():
            vocab[k] = Counter()
            vocab[k]['<UNK>'] = 1000000

        for f in files:
            if f is None:
                continue
            df = pd.read_csv(f, sep=self.sep, header=self.header)
            for label, text in zip(df[self.label], df[self.text]):
                text = ' '.join(list(filter(lambda s: len(s) != 0, [self.clean_fn(w) for w in text.split()])))
                text = list(filter(lambda s: len(s) != 0, re.split('\s+', text)))
                if not text:
                    continue

                for k, vectorizer in self.vectorizers.items():
                    vocab_file = vectorizer.count(text)
                    vocab[k].update(vocab_file)

                if label not in self.label2index:
                    self.label2index[label] = label_idx
                    label_idx += 1

        return vocab, self.get_labels()

    def load(self, filename, vocabs, batchsz, **kwargs):
        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []
        df = pd.read_csv(filename, sep=self.sep, header=self.header)
        for label, text in zip(df[self.label], df[self.text]):
            text = ' '.join(list(filter(lambda s: len(s) != 0, [self.clean_fn(w) for w in text.split()])))
            text = list(filter(lambda s: len(s) != 0, re.split('\s+', text)))
            if not text:
                continue
            y = self.label2index[label]
            example_dict = dict()
            for k, vectorizer in self.vectorizers.items():
                example_dict[k], lengths = vectorizer.run(text, vocabs[k])
                if lengths is not None:
                    example_dict['{}_lengths'.format(k)] = lengths

            example_dict['y'] = y
            examples.append(example_dict)
        return baseline.data.ExampleDataFeed(baseline.data.DictExamples(examples,
                                                                        do_shuffle=shuffle,
                                                                        sort_key=sort_key),
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim)


