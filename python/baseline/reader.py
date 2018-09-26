import baseline.data
import numpy as np
from collections import Counter
import re
import codecs
from baseline.utils import import_user_module, revlut, export
from baseline.vectorizers import Dict1DVectorizer, GOVectorizer
import os

__all__ = []
exporter = export(__all__)


@exporter
def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for _ in f:
            lines = lines + 1
    return lines


def _build_vocab_for_col(col, files, vectorizers):
    vocabs = dict()

    for key in vectorizers.keys():
        vocabs[key] = Counter()
        vocabs[key]['<UNK>'] = 100000  # In case freq cutoffs
        vocabs[key]['<EOS>'] = 100000  # In case freq cutoffs
    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                cols = re.split("\t", line)
                text = re.split("\s", cols[col])
                for key, vectorizer in vectorizers.items():
                    vocabs[key].update(vectorizer.count(text))
    return vocabs


@exporter
class ParallelCorpusReader(object):

    def __init__(self,
                 vectorizers,
                 trim=False):

        self.src_vectorizers = {}
        self.tgt_vectorizer = None
        for k, vectorizer in vectorizers.items():
            if k == 'tgt':
                self.tgt_vectorizer = GOVectorizer(vectorizer)
            else:
                self.src_vectorizers[k] = vectorizer
        self.trim = trim

    def build_vocabs(self, files):
        pass

    def load_examples(self, tsfile, vocab1, vocab2):
        pass

    def load(self, tsfile, vocab1, vocab2, batchsz, shuffle=False, sort_key=None):
        examples = self.load_examples(tsfile, vocab1, vocab2)
        return baseline.data.ExampleDataFeed(examples, batchsz,
                                             shuffle=shuffle, trim=self.trim, sort_key=sort_key)


@exporter
class TSVParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, vectorizers,
                 trim=False, src_col_num=0, tgt_col_num=1):
        super(TSVParallelCorpusReader, self).__init__(vectorizers, trim)
        self.src_col_num = src_col_num
        self.tgt_col_num = tgt_col_num

    def build_vocabs(self, files):
        src_vocab = _build_vocab_for_col(self.src_col_num, files, self.src_vectorizers)
        tgt_vocab = _build_vocab_for_col(self.tgt_col_num, files, {'tgt': self.tgt_vectorizer})
        return src_vocab, tgt_vocab['tgt']

    def load_examples(self, tsfile, src_vocabs, tgt_vocab):
        ts = []
        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                splits = re.split("\t", line.strip())
                src = list(filter(lambda x: len(x) != 0, re.split("\s+", splits[0])))

                example = {}
                for k, vectorizer in self.src_vectorizers.items():
                    example[k], length = vectorizer.run(src, src_vocabs[k])
                    if length is not None:
                        example['{}_lengths'.format(k)] = length

                tgt = list(filter(lambda x: len(x) != 0, re.split("\s+", splits[1])))
                example['tgt'], example['tgt_lengths'] = self.tgt_vectorizer.run(tgt, tgt_vocab)
                ts += [example]
        return baseline.data.Seq2SeqExamples(ts)


@exporter
class MultiFileParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, src_suffix, tgt_suffix, vectorizers, trim=False):
        super(MultiFileParallelCorpusReader, self).__init__(vectorizers, trim)
        self.src_suffix = src_suffix
        self.tgt_suffix = tgt_suffix
        if not src_suffix.startswith('.'):
            self.src_suffix = '.' + self.src_suffix
        if not tgt_suffix.startswith('.'):
            self.tgt_suffix = '.' + self.tgt_suffix

    # 2 possibilities here, either we have a vocab file, e.g. vocab.bpe.32000, or we are going to generate
    # from each column
    def build_vocabs(self, files):
        if len(files) == 1 and os.path.exists(files[0]):
            src_vocab = _build_vocab_for_col(0, files, self.src_vectorizers)
            tgt_vocab = _build_vocab_for_col(0, files, {'tgt': self.tgt_vectorizer})
        else:
            src_vocab = _build_vocab_for_col(0, [f + self.src_suffix for f in files], self.src_vectorizers)
            tgt_vocab = _build_vocab_for_col(0, [f + self.tgt_suffix for f in files], {'tgt': self.tgt_vectorizer})
        return src_vocab, tgt_vocab['tgt']

    def load_examples(self, tsfile, src_vocabs, tgt_vocab):
        ts = []

        with codecs.open(tsfile + self.src_suffix, encoding='utf-8', mode='r') as fsrc:
            with codecs.open(tsfile + self.tgt_suffix, encoding='utf-8', mode='r') as ftgt:
                for src, tgt in zip(fsrc, ftgt):
                    example = {}
                    src = re.split("\s+", src.strip())
                    for k, vectorizer in self.src_vectorizers.items():
                        example[k], length = vectorizer.run(src, src_vocabs[k])
                        if length is not None:
                            example['{}_lengths'.format(k)] = length
                    tgt = re.split("\s+", tgt.strip())
                    example['tgt'], example['tgt_lengths'] = self.tgt_vectorizer.run(tgt, tgt_vocab)
                    ts += [example]
        return baseline.data.Seq2SeqExamples(ts)


@exporter
def create_parallel_corpus_reader(vectorizers, trim, **kwargs):

    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        print('Reading parallel file corpus')
        pair_suffix = kwargs.get('pair_suffix')
        reader = MultiFileParallelCorpusReader(pair_suffix[0], pair_suffix[1], vectorizers, trim)
    elif reader_type == 'tsv':
        print('Reading tab-separated corpus')
        reader = TSVParallelCorpusReader(vectorizers, trim)
    else:
        mod = import_user_module("reader", reader_type)
        return mod.create_parallel_corpus_reader(vectorizers, trim, **kwargs)
    return reader


@exporter
class SeqPredictReader(object):

    def __init__(self,
                 vectorizers,
                 trim=False):
        self.vectorizers = vectorizers
        self.trim = trim
        self.label2index = {"<PAD>": 0, "<GO>": 1, "<EOS>": 2}
        self.label_vectorizer = Dict1DVectorizer(fields='y', mxlen=-1)

    def build_vocab(self, files):

        vocabs = {}
        for key in self.vectorizers.keys():
            vocabs[key] = Counter()
            vocabs[key]['<UNK>'] = 100000  # In case freq cutoffs

        labels = Counter()
        for file in files:
            if file is None:
                continue

            examples = self.read_examples(file)
            for example in examples:
                labels.update(self.label_vectorizer.count(example))
                for k, vectorizer in self.vectorizers.items():
                    vocab_example = vectorizer.count(example)
                    vocabs[k].update(vocab_example)

        base_offset = len(self.label2index)
        for i, k in enumerate(labels.keys()):
            self.label2index[k] = i + base_offset
        return vocabs

    def read_examples(self):
        pass

    def load(self, filename, vocabs, batchsz, shuffle=False, sort_key=None):

        ts = []
        texts = self.read_examples(filename)

        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        for i, example_tokens in enumerate(texts):
            example = {}
            for k, vectorizer in self.vectorizers.items():
                example[k], lengths = vectorizer.run(example_tokens, vocabs[k])
                if lengths is not None:
                    example['{}_lengths'.format(k)] = lengths
            example['y'], lengths = self.label_vectorizer.run(example_tokens, self.label2index)
            example['y_lengths'] = lengths
            example['ids'] = i
            ts.append(example)
        examples = baseline.data.DictExamples(ts, do_shuffle=shuffle, sort_key=sort_key)
        return baseline.data.ExampleDataFeed(examples, batchsz=batchsz, shuffle=shuffle, trim=self.trim), texts


@exporter
class CONLLSeqReader(SeqPredictReader):

    def __init__(self, vectorizers, trim=False, **kwargs):
        super(CONLLSeqReader, self).__init__(vectorizers, trim)
        self.named_fields = kwargs.get('named_fields', {})

    def read_examples(self, tsfile):

        tokens = []
        examples = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                states = re.split("\s", line.strip())

                token = dict()
                if len(states) > 1:
                    for j in range(len(states)):
                        noff = j - len(states)
                        if noff >= 0:
                            noff = j
                        field_name = self.named_fields.get(str(j),
                                                           self.named_fields.get(str(noff), str(j)))
                        token[field_name] = states[j]
                    tokens += [token]

                else:
                    examples += [tokens]
                    tokens = []

        return examples


@exporter
def create_seq_pred_reader(vectorizers, trim, **kwargs):

    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        print('Reading CONLL sequence file corpus')
        reader = CONLLSeqReader(vectorizers, trim, **kwargs)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_seq_pred_reader(vectorizers, trim, **kwargs)
    return reader


@exporter
class SeqLabelReader(object):

    def __init__(self):
        pass

    def build_vocab(self, files, **kwargs):
        pass

    def load(self, filename, index, batchsz, **kwargs):
        pass


@exporter
class TSVSeqLabelReader(SeqLabelReader):

    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }

    def __init__(
            self,
            vectorizers, clean_fn=None, trim=False
    ):
        super(TSVSeqLabelReader, self).__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = clean_fn
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim

    SPLIT_ON = '[\t\s]+'

    @staticmethod
    def splits(text):
        return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

    @staticmethod
    def do_clean(l):
        l = l.lower()
        l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
        for k, v in TSVSeqLabelReader.REPLACE.items():
            l = l.replace(k, v)
        return l.strip()

    @staticmethod
    def label_and_sentence(line, clean_fn):
        label_text = re.split(TSVSeqLabelReader.SPLIT_ON, line)
        label = label_text[0]
        text = label_text[1:]
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text])))
        text = list(filter(lambda s: len(s) != 0, re.split('\s+', text)))
        return label, text

    def build_vocab(self, files, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary
        
        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.
        
        :param files: Either a directory (str), or an array of individual files
        :return: 
        """
        label_idx = len(self.label2index)
        if type(files) == str:
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]
        vocab = dict()
        for k in self.vectorizers.keys():
            vocab[k] = Counter()
            vocab[k]['<UNK>'] = 100000  # In case freq cutoffs

        for file in files:
            if file is None:
                continue
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for il, line in enumerate(f):
                    label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
                    if len(text) == 0:
                        continue

                    for k, vectorizer in self.vectorizers.items():
                        vocab_file = vectorizer.count(text)
                        vocab[k].update(vocab_file)

                    if label not in self.label2index:
                        self.label2index[label] = label_idx
                        label_idx += 1

        return vocab, self.get_labels()

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load(self, filename, vocabs, batchsz, **kwargs):

        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []

        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for il, line in enumerate(f):
                label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
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
                                             batchsz=batchsz, shuffle=shuffle, trim=self.trim)


@exporter
def create_pred_reader(vectorizers, clean_fn, **kwargs):
    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        trim = kwargs.get('trim', False)
        #splitter = kwargs.get('splitter', '[\t\s]+')
        reader = TSVSeqLabelReader(vectorizers, clean_fn=clean_fn, trim=trim)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_pred_reader(vectorizers, clean_fn=clean_fn, **kwargs)
    return reader


@exporter
class LineSeqReader(object):

    def __init__(self, vectorizers, nbptt):
        self.nbptt = nbptt
        self.vectorizers = vectorizers

    def build_vocab(self, files):

        vocabs = {}
        for key in self.vectorizers.keys():
            vocabs[key] = Counter()
            vocabs[key] = Counter()
            vocabs[key]['<UNK>'] = 100000  # In case freq cutoffs
        for file in files:
            if file is None:
                continue

            with codecs.open(file, encoding='utf-8', mode='r') as f:
                sentences = []
                for line in f:
                    sentences += line.split() + ['<EOS>']
                for k, vectorizer in self.vectorizers.items():
                    vocabs[k].update(vectorizer.count(sentences))
        return vocabs

    def load(self, filename, vocabs, batchsz, tgt_key='x'):

        x = dict()
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            sentences = []
            for line in f:
                sentences += line.split() + ['<EOS>']
            for k, vectorizer in self.vectorizers.items():
                vec, valid_lengths = vectorizer.run(sentences, vocabs[k])
                x[k] = vec[:valid_lengths]
                shp = list(vectorizer.get_dims())
                shp[0] = valid_lengths
                x['{}_dims'.format(k)] = tuple(shp)

        return baseline.data.SeqWordCharDataFeed(x, self.nbptt, batchsz, tgt_key=tgt_key)



@exporter
def create_lm_reader(vectorizers, nbptt, **kwargs):
    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        reader = LineSeqReader(vectorizers, nbptt)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_lm_reader(vectorizers, nbptt, **kwargs)
    return reader
