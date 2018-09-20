import io
import contextlib
import numpy as np
from baseline.utils import export, load_user_embeddings, write_json, read_config_file
from baseline.mime_type import mime_type
__all__ = []
exporter = export(__all__)


def norm_weights(word_vectors):
    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    norms = (norms == 0) + norms
    return word_vectors / norms


@exporter
class EmbeddingsModel(object):
    def __init__(self):
        super(EmbeddingsModel, self).__init__()

    def get_dsz(self):
        pass

    def get_vsz(self):
        pass

    def get_vocab(self):
        pass

    def save_md(self, target):
        pass


def pool_vec(embeddings, tokens, operation=np.mean):
    if type(tokens) is str:
        tokens = tokens.split()
    try:
        return operation([embeddings.lookup(t, False) for t in tokens], 0)
    except:
        return embeddings.weights[0]


@exporter
class WordEmbeddingsModel(object):
    def __init__(self, **kwargs):
        super(WordEmbeddingsModel, self).__init__()
        self.vocab = kwargs.get('vocab')
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.weights = kwargs.get('weights')
        if 'known_vocab_file' in kwargs:
            md = read_config_file(kwargs['md_file'])
            self.vocab = md['vocab']
        if 'weights_file' in kwargs:
            self.weights = np.load(kwargs['weights_file']).get('arr_0')

        if self.weights is not None:
            if self.vsz is None:
                self.vsz = self.weights.shape[0]
            else:
                assert self.vsz == self.weights.shape[0]
            if self.dsz is None:
                self.dsz = self.weights.shape[1]
            else:
                assert self.dsz == self.weights.shape[1]

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def get_vocab(self):
        return self.vocab

    def get_weights(self):
        return self.weights

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz(), 'vocab': self.get_vocab()}, target)

    def save_weights(self, target):
        np.savez(target, self.weights)

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

    def __getitem__(self, word):
        return self.lookup(word, nullifabsent=False)


@exporter
class PretrainedEmbeddingsModel(WordEmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False, normalize=False, **kwargs):
        super(PretrainedEmbeddingsModel, self).__init__()

        if known_vocab is None and keep_unused is False:
            print('Warning: known_vocab=None, keep_unused=False. Setting keep_unused=True, all vocab will be preserved')
            keep_unused = True
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        idx = 1

        word_vectors, self.dsz, known_vocab, idx = self._read_vectors(filename, idx, known_vocab, keep_unused, **kwargs)
        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        word_vectors = [self.nullv] + word_vectors
        self.vocab["<PAD>"] = 0

        if known_vocab is not None:
            unknown = {v: cnt for v, cnt in known_vocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        self.weights = np.array(word_vectors)
        if normalize is True:
            self.weights = norm_weights(self.weights)

        self.vsz = self.weights.shape[0]

    def _read_vectors(self, filename, idx, known_vocab, keep_unused, **kwargs):
        use_mmap = bool(kwargs.get('use_mmap', False))
        read_fn = self._read_word2vec_file
        is_glove_file = mime_type(filename) == 'text/plain'
        if use_mmap:
            if is_glove_file:
                read_fn = self._read_glove_mmap
            else:
                read_fn = self._read_word2vec_mmap
        elif is_glove_file:
            read_fn = self._read_glove_file

        return read_fn(filename, idx, known_vocab, keep_unused)

    def _read_word2vec_file(self, filename, idx, known_vocab, keep_unused):
        word_vectors = []
        with io.open(filename, "rb") as f:
            header = f.readline()
            vsz, dsz = map(int, header.split())
            width = 4 * dsz
            for i in range(vsz):
                word = self._readtospc(f)
                raw = f.read(width)
                if keep_unused is False and word not in known_vocab:
                    continue
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0
                vec = np.fromstring(raw, dtype=np.float32)
                word_vectors.append(vec)
                self.vocab[word] = idx
                idx += 1
        return word_vectors, dsz, known_vocab, idx

    @staticmethod
    def _read_word2vec_line_mmap(m, width, start):
        current = start+1
        while m[current:current+1] != b' ':
            current += 1
        vocab = m[start:current].decode('utf-8')
        raw = m[current+1:current+width+1]
        value = np.fromstring(raw, dtype=np.float32)
        return vocab, value, current+width + 1

    def _read_word2vec_mmap(self, filename, idx, known_vocab, keep_unused):
        import mmap
        word_vectors = []
        with io.open(filename, 'rb') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                header_end = m[:50].find(b'\n')
                vsz, dsz = map(int, (m[:header_end]).split(b' '))
                width = 4 * dsz
                current = header_end + 1
                for i in range(vsz):
                    word, vec, current = self._read_word2vec_line_mmap(m, width, current)
                    if keep_unused is False and word not in known_vocab:
                        continue
                    if known_vocab and word in known_vocab:
                        known_vocab[word] = 0

                    word_vectors.append(vec)
                    self.vocab[word] = idx
                    idx += 1
                return word_vectors, dsz, known_vocab, idx

    @staticmethod
    def _readtospc(f):

        s = bytearray()
        ch = f.read(1)

        while ch != b' ':
            s.extend(ch)
            ch = f.read(1)
        s = s.decode('utf-8')
        # Only strip out normal space and \n not other spaces which are words.
        return s.strip(' \n')

    def _read_glove_file(self, filename, idx, known_vocab, keep_unused):
        word_vectors = []
        with io.open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                values = line.split(" ")
                word = values[0]
                if word in self.vocab: continue
                if keep_unused is False and word not in known_vocab or word in self.vocab:
                    continue
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0
                vec = np.asarray(values[1:], dtype=np.float32)
                word_vectors.append(vec)
                self.vocab[word] = idx
                idx += 1
        dsz = vec.shape[0]
        return word_vectors, dsz, known_vocab, idx

    def _read_glove_mmap(self, filename, idx, known_vocab, keep_unused):
        import mmap
        word_vectors = []
        with io.open(filename, "r", encoding="utf-8") as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                for line in iter(m.readline, b''):
                    line = line.rstrip(b"\n")
                    values = line.split(b" ")
                    if len(values) == 0:
                        break
                    word = values[0].decode('utf-8')
                    if word in self.vocab: continue
                    if keep_unused is False and word not in known_vocab:
                        continue
                    if known_vocab and word in known_vocab:
                        known_vocab[word] = 0
                    vec = np.asarray(values[1:], dtype=np.float32)
                    word_vectors.append(vec)
                    self.vocab[word] = idx
                    idx += 1
                dsz = vec.shape[0]
                return word_vectors, dsz, known_vocab, idx


@exporter
class RandomInitVecModel(EmbeddingsModel):

    def __init__(self, dsz, known_vocab, counts=True, unif_weight=None):
        super(RandomInitVecModel, self).__init__()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = dict()
        self.vocab["<PAD>"] = 0
        self.dsz = dsz
        self.vsz = 1

        if counts is True:
            attested = [v for v, cnt in known_vocab.items() if cnt > 0]
            for k, v in enumerate(attested):
                self.vocab[v] = k + 1
                self.vsz += 1
        else:
            print('Restoring existing vocab')
            self.vocab = known_vocab
            self.vsz = len(self.vocab)

        self.weights = np.random.uniform(-uw, uw, (self.vsz, self.dsz))

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz(), 'vocab': self.get_vocab()}, target)


@exporter
def load_embeddings(filename, known_vocab=None, **kwargs):
    embed_type = kwargs.get('embed_type', 'default')
    if embed_type == 'default':
        return PretrainedEmbeddingsModel(filename,
                                         known_vocab=known_vocab,
                                         unif_weight=kwargs.pop('unif', 0),
                                         keep_unused=kwargs.pop('keep_unused', False),
                                         normalize=kwargs.pop('normalized', False), **kwargs)
    print('loading user module')
    return load_user_embeddings(filename, known_vocab, **kwargs)