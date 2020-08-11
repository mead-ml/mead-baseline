import re
import codecs
from collections import Counter
import baseline
from baseline.utils import listify
from baseline.reader import register_reader, SeqLabelReader, _filter_vocab, _norm_ext


@register_reader(task='classify', name='parallel')
class ParallelSeqLabelReader(SeqLabelReader):
    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = kwargs.get('clean_fn')
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim
        self.truncate = truncate
        self.data = _norm_ext(kwargs.get('data_suffix', kwargs.get('data', 'in')))
        self.labels = _norm_ext(kwargs.get('label_suffix', kwargs.get('labels', 'labels')))


    SPLIT_ON = '[\t\s]+'

    @staticmethod
    def splits(text):
        return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    @staticmethod
    def do_clean(l):
        l = l.lower()
        l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
        for k, v in ParallelSeqLabelReader.REPLACE.items():
            l = l.replace(k, v)
        return l.strip()

    @staticmethod
    def get_sentence(line, clean_fn):
        text = re.split(ParallelSeqLabelReader.SPLIT_ON, line)
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text])))
        text = list(filter(lambda s: len(s) != 0, re.split('\s+', text)))
        return text

    def build_vocab(self, files, **kwargs):
        label_idx = len(self.label2index)
        files = listify(files)
        vocab = {k: Counter() for k in self.vectorizers.keys()}
        for file_name in files:
            if file_name is None: continue
            with codecs.open(file_name + self.data, encoding='utf-8', mode='r') as data_file:
                with codecs.open(file_name + self.labels, encoding='utf-8', mode='r') as label_file:
                    for d, l in zip(data_file, label_file):
                        if d.strip() == "": continue
                        label = l.rstrip()
                        text = ParallelSeqLabelReader.get_sentence(d, self.clean_fn)
                        if len(text) == 0: continue

                        for k, vectorizer in self.vectorizers.items():
                            vocab_file = vectorizer.count(text)
                            vocab[k].update(vocab_file)

                        if label not in self.label2index:
                            self.label2index[label] = label_idx
                            label_idx += 1

        vocab = _filter_vocab(vocab, kwargs.get('min_f', {}))

        return vocab, self.get_labels()

    def load(self, filename, vocabs, batchsz, **kwargs):
        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []

        with codecs.open(filename + self.data, encoding='utf-8', mode='r') as data_f:
            with codecs.open(filename + self.labels, encoding='utf-8', mode='r') as label_f:
                for d, l in zip(data_f, label_f):
                    if d.strip() == "": continue
                    label = l.rstrip()
                    text = ParallelSeqLabelReader.get_sentence(d, self.clean_fn)
                    if len(text) == 0:
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
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim, truncate=self.truncate)
