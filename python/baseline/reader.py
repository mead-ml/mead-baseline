import baseline.data
import numpy as np
from collections import Counter
import re
import codecs
from baseline.utils import import_user_module, revlut
import os


def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for _ in f:
            lines = lines + 1
    return lines


def _build_vocab_for_col(col, files):
    vocab = Counter()
    vocab['<GO>'] = 1
    vocab['<EOS>'] = 1

    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                cols = re.split("\t", line)
                text = re.split("\s", cols[col])

                for w in text:
                    w = w.strip()
                    vocab[w] += 1
    return vocab


class ParallelCorpusReader(object):

    def __init__(self,
                 max_sentence_length=1000,
                 vec_alloc=np.zeros,
                 src_vec_trans=None,
                 trim=False):
        self.vec_alloc = vec_alloc
        self.src_vec_trans = src_vec_trans
        self.max_sentence_length = max_sentence_length
        self.trim = trim

    def build_vocabs(self, files):
        pass

    def load_examples(self, tsfile, vocab1, vocab2):
        pass

    def load(self, tsfile, vocab1, vocab2, batchsz, shuffle=False):
        examples = self.load_examples(tsfile, vocab1, vocab2)
        return baseline.data.Seq2SeqDataFeed(examples, batchsz,
                                             shuffle=shuffle, src_vec_trans=self.src_vec_trans,
                                             vec_alloc=self.vec_alloc, trim=self.trim)


class TSVParallelCorpusReader(ParallelCorpusReader):

    def __init__(self,
                 max_sentence_length=1000,
                 vec_alloc=np.zeros,
                 src_vec_trans=None,
                 trim=False, src_col_num=0, dst_col_num=1):
        super(TSVParallelCorpusReader, self).__init__(max_sentence_length, vec_alloc, src_vec_trans, trim)
        self.src_col_num = src_col_num
        self.dst_col_num = dst_col_num

    def build_vocabs(self, files):
        src_vocab = _build_vocab_for_col(self.src_col_num, files)
        dst_vocab = _build_vocab_for_col(self.dst_col_num, files)
        return src_vocab, dst_vocab

    def load_examples(self, tsfile, vocab1, vocab2):
        GO = vocab2['<GO>']
        EOS = vocab2['<EOS>']
        mxlen = self.max_sentence_length
        ts = []
        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                splits = re.split("\t", line.strip())
                src = re.split("\s+", splits[0])
                dst = re.split("\s+", splits[1])

                srcl = self.vec_alloc(mxlen, dtype=np.int)
                tgtl = self.vec_alloc(mxlen, dtype=np.int)
                src_len = len(src)
                tgt_len = len(dst) + 2  # <GO>,...,<EOS>
                end1 = min(src_len, mxlen)
                end2 = min(tgt_len, mxlen)-2
                tgtl[0] = GO
                src_len = end1
                tgt_len = end2+2

                for j in range(end1):
                    srcl[j] = vocab1[src[j]]
                for j in range(end2):
                    tgtl[j + 1] = vocab2[dst[j]]

                tgtl[end2] = EOS

                ts.append((srcl, tgtl, src_len, tgt_len))

        return baseline.data.Seq2SeqExamples(ts)


class MultiFileParallelCorpusReader(ParallelCorpusReader):

    def __init__(self, src_suffix, dst_suffix,
                 max_sentence_length=1000,
                 vec_alloc=np.zeros,
                 src_vec_trans=None,
                 trim=False):
        super(MultiFileParallelCorpusReader, self).__init__(max_sentence_length, vec_alloc, src_vec_trans, trim)
        self.src_suffix = src_suffix
        self.dst_suffix = dst_suffix
        if not src_suffix.startswith('.'):
            self.src_suffix = '.' + self.src_suffix
        if not dst_suffix.startswith('.'):
            self.dst_suffix = '.' + self.dst_suffix

    def build_vocabs(self, files):
        src_vocab = _build_vocab_for_col(0, files)
        dst_vocab = src_vocab
        return src_vocab, dst_vocab

    def load_examples(self, tsfile, vocab1, vocab2):
        PAD = vocab1['<PAD>']
        GO = vocab2['<GO>']
        EOS = vocab2['<EOS>']
        mxlen = self.max_sentence_length
        ts = []

        with codecs.open(tsfile + self.src_suffix, encoding='utf-8', mode='r') as fsrc:
            with codecs.open(tsfile + self.dst_suffix, encoding='utf-8', mode='r') as fdst:
                for src, dst in zip(fsrc, fdst):

                    src = re.split("\s+", src.strip())
                    dst = re.split("\s+", dst.strip())
                    srcl = self.vec_alloc(mxlen, dtype=np.int)
                    tgtl = self.vec_alloc(mxlen, dtype=np.int)
                    src_len = len(src)
                    tgt_len = len(dst) + 2
                    end1 = min(src_len, mxlen)
                    end2 = min(tgt_len, mxlen)-2
                    tgtl[0] = GO
                    src_len = end1
                    tgt_len = end2+2

                    for j in range(end1):
                        srcl[j] = vocab1[src[j]]
                    for j in range(end2):
                        tgtl[j + 1] = vocab2[dst[j]]

                    tgtl[end2] = EOS
                    ts.append((srcl, tgtl, src_len, tgt_len))

        return baseline.data.Seq2SeqExamples(ts)


def create_parallel_corpus_reader(mxlen, alloc_fn, trim, src_vec_trans, **kwargs):

    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        print('Reading parallel file corpus')
        pair_suffix = kwargs.get('pair_suffix')
        reader = MultiFileParallelCorpusReader(pair_suffix[0], pair_suffix[1],
                                               mxlen, alloc_fn,
                                               src_vec_trans, trim)
    elif reader_type == 'tsv':
        print('Reading tab-separated corpus')
        reader = TSVParallelCorpusReader(mxlen, alloc_fn, src_vec_trans, trim)
    else:
        mod = import_user_module("reader", reader_type)
        return mod.create_parallel_corpus_reader(mxlen, alloc_fn,
                                                 src_vec_trans, trim, **kwargs)
    return reader


def identity_trans_fn(x):
    return x


class CONLLSeqReader(object):

    UNREP_EMOTICONS = (
        ':)',
        ':(((',
        ':D',
        '=)',
        ':-)',
        '=(',
        '(=',
        '=[[',
    )

    def __init__(self, max_sentence_length=-1, max_word_length=-1, word_trans_fn=None,
                 vec_alloc=np.zeros, vec_shape=np.shape, trim=False):
        self.cleanup_fn = identity_trans_fn if word_trans_fn is None else word_trans_fn
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.vec_alloc = vec_alloc
        self.vec_shape = vec_shape
        self.trim = trim
        self.label2index = {"<PAD>": 0, "<GO>": 1, "<EOS>": 2}

    @staticmethod
    def web_cleanup(word):
        if word.startswith('http'): return 'URL'
        if word.startswith('@'): return '@@@@'
        if word.startswith('#'): return '####'
        if word == '"': return ','
        if word in CONLLSeqReader.UNREP_EMOTICONS: return ';)'
        if word == '<3': return '&lt;3'
        return word

    def build_vocab(self, files):
        vocab_word = Counter()
        vocab_ch = Counter()
        maxw = 0
        maxs = 0
        for file in files:
            if file is None:
                continue

            sl = 0
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for line in f:

                    line = line.strip()
                    if line == '':
                        maxs = max(maxs, sl)
                        sl = 0

                    else:
                        states = re.split("\s", line)
                        sl += 1
                        w = states[0]
                        vocab_word[self.cleanup_fn(w)] += 1
                        maxw = max(maxw, len(w))
                        for k in w:
                            vocab_ch[k] += 1

        self.max_word_length = min(maxw, self.max_word_length) if self.max_word_length > 0 else maxw
        self.max_sentence_length = min(maxs, self.max_sentence_length) if self.max_sentence_length > 0 else maxs
        print('Max sentence length %d' % self.max_sentence_length)
        print('Max word length %d' % self.max_word_length)

        return vocab_ch, vocab_word

    @staticmethod
    def read_lines(tsfile):

        txts = []
        lbls = []
        txt = []
        lbl = []

        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                states = re.split("\s", line.strip())

                if len(states) > 1:
                    txt.append(states[0])
                    lbl.append(states[-1])
                else:
                    txts.append(txt)
                    lbls.append(lbl)
                    txt = []
                    lbl = []

        return txts, lbls

    def load(self, filename, words_vocab, chars_vocab, batchsz, shuffle=False):

        ts = []
        idx = 2 # GO=0, START=1, EOS=2
        mxlen = self.max_sentence_length
        maxw = self.max_word_length
        txts, lbls = CONLLSeqReader.read_lines(filename)

        for i in range(len(txts)):

            xs_ch = self.vec_alloc((mxlen, maxw), dtype=np.int)
            xs = self.vec_alloc((mxlen), dtype=np.int)
            ys = self.vec_alloc((mxlen), dtype=np.int)

            lv = lbls[i]
            v = txts[i]

            length = mxlen
            for j in range(mxlen):

                if j == len(v):
                    length = j
                    break

                w = v[j]
                nch = min(len(w), maxw)
                label = lv[j]

                if label not in self.label2index:
                    idx += 1
                    self.label2index[label] = idx

                ys[j] = self.label2index[label]
                xs[j] = words_vocab.get(self.cleanup_fn(w))
                for k in range(nch):
                    xs_ch[j, k] = chars_vocab.get(w[k], 0)

            ts.append((xs, xs_ch, ys, length, i))
        examples = baseline.data.SeqWordCharTagExamples(ts)
        return baseline.data.SeqWordCharLabelDataFeed(examples, batchsz=batchsz, shuffle=shuffle, vec_alloc=self.vec_alloc, vec_shape=self.vec_shape), txts


def create_seq_pred_reader(mxlen, mxwlen, word_trans_fn, vec_alloc, vec_shape, trim, **kwargs):

    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        print('Reading CONLL sequence file corpus')
        reader = CONLLSeqReader(mxlen, mxwlen, word_trans_fn,
                                vec_alloc, vec_shape, trim)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_seq_pred_reader(mxlen, mxwlen, word_trans_fn,
                                            vec_alloc, vec_shape, trim, **kwargs)
    return reader


class SeqLabelReader(object):

    def __init__(self):
        pass

    def build_vocab(self, files, **kwargs):
        pass

    def load(self, filename, index, batchsz, **kwargs):
        pass


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

    def __init__(self, mxlen=1000, mxfiltsz=0, clean_fn=None, vec_alloc=np.zeros, src_vec_trans=None):
        super(TSVSeqLabelReader, self).__init__()

        self.vocab = None
        self.label2index = {}
        self.clean_fn = clean_fn
        self.mxlen = mxlen
        self.mxfiltsz = mxfiltsz
        self.vec_alloc = vec_alloc
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.src_vec_trans = src_vec_trans

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
        label_text = re.split('[\t\s]+', line)
        label = label_text[0]
        text = label_text[1:]
        text = ' '.join(list(filter(lambda s: len(s) != 0, [clean_fn(w) for w in text])))
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

        vocab = Counter()
        for file in files:
            if file is None:
                continue
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for line in f:
                    label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
                    if label not in self.label2index:
                        self.label2index[label] = label_idx
                        label_idx += 1
                    for w in TSVSeqLabelReader.splits(text):
                        vocab[w] += 1

        return vocab, self.get_labels()

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load(self, filename, index, batchsz, **kwargs):

        PAD = index['<PAD>']
        shuffle = kwargs.get('shuffle', False)
        halffiltsz = self.mxfiltsz // 2
        nozplen = self.mxlen - 2*halffiltsz

        examples = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for offset, line in enumerate(f):
                label, text = TSVSeqLabelReader.label_and_sentence(line, self.clean_fn)
                y = self.label2index[label]
                toks = TSVSeqLabelReader.splits(text)
                mx = min(len(toks), nozplen)
                toks = toks[:mx]
                x = self.vec_alloc(self.mxlen, dtype=int)
                for j in range(len(toks)):
                    w = toks[j]
                    key = index.get(w, PAD)
                    x[j+halffiltsz] = key
                examples.append((x, y))
        return baseline.data.SeqLabelDataFeed(baseline.data.SeqLabelExamples(examples),
                                              batchsz=batchsz, shuffle=shuffle, vec_alloc=self.vec_alloc, src_vec_trans=self.src_vec_trans)


def create_pred_reader(mxlen, zeropadding, clean_fn, vec_alloc, src_vec_trans, **kwargs):
    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        reader = TSVSeqLabelReader(mxlen, zeropadding, clean_fn, vec_alloc, src_vec_trans)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_pred_reader(mxlen, zeropadding, clean_fn, vec_alloc, src_vec_trans, **kwargs)
    return reader


class PTBSeqReader(object):

    def __init__(self, max_word_length, nbptt):
        self.max_word_length = max_word_length
        self.nbptt = nbptt

    def build_vocab(self, files):
        vocab_word = Counter()
        vocab_ch = Counter()
        maxw = 0
        num_words_in_files = []
        for file in files:
            if file is None:
                continue

            with codecs.open(file, encoding='utf-8', mode='r') as f:
                num_words = 0
                for line in f:
                    sentence = line.split() + ['<EOS>']
                    num_words += len(sentence)
                    for w in sentence:
                        vocab_word[w] += 1
                        maxw = max(maxw, len(w))
                        for k in w:
                            vocab_ch[k] += 1
                num_words_in_files.append(num_words)

        self.max_word_length = min(maxw, self.max_word_length)
        print('Max word length %d' % self.max_word_length)

        return vocab_ch, vocab_word, num_words_in_files

    def load(self, filename, words_vocab, chars_vocab, num_words, batchsz, vec_alloc=np.zeros):

        xch = vec_alloc((num_words, self.max_word_length), np.int)
        x = vec_alloc((num_words), np.int)
        i = 0
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for line in f:
                sentence = line.split() + ['<EOS>']
                num_words += len(sentence)
                for w in sentence:
                    x[i] = words_vocab.get(w)
                    nch = min(len(w), self.max_word_length)
                    for k in range(nch):
                        xch[i, k] = chars_vocab.get(w[k], 0)
                    i += 1

        return baseline.data.SeqWordCharDataFeed(x, xch, self.nbptt, batchsz, self.max_word_length)


def create_lm_reader(max_word_length, nbptt, **kwargs):
    reader_type = kwargs.get('reader_type', 'default')

    if reader_type == 'default':
        reader = PTBSeqReader(max_word_length, nbptt)
    else:
        mod = import_user_module("reader", reader_type)
        reader = mod.create_lm_reader(max_word_length, nbptt, **kwargs)
    return reader
