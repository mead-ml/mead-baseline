import io
import contextlib
import numpy as np
from baseline.utils import export

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
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

    def mean_vec(self, tokens):
        if type(tokens) is str:
            tokens = tokens.split()
        try:
            return np.mean([self.lookup(t, False) for t in tokens], 0)
        except:
            return self.weights[0]


@exporter
class PretrainedEmbeddingsModel(EmbeddingsModel):

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

        self.vsz = self.weights.shape[0] - 1

    def _read_vectors(self, filename, idx, known_vocab, keep_unused, **kwargs):
        pass


@exporter
class Word2VecModel(PretrainedEmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False, normalize=False, **kwargs):
        super(Word2VecModel, self).__init__(filename, known_vocab, unif_weight, keep_unused, normalize, **kwargs)

    def _read_vectors(self, filename, idx, known_vocab, keep_unused, **kwargs):
        use_mmap = bool(kwargs.get('use_mmap', False))

        read_fn = self._read_vectors_mmap if use_mmap else self._read_vectors_file
        return read_fn(filename, idx, known_vocab, keep_unused)

    def _read_vectors_file(self, filename, idx, known_vocab, keep_unused):
        word_vectors = []
        with io.open(filename, "rb") as f:
            header = f.readline()
            vsz, dsz = map(int, header.split())
            width = 4 * dsz
            for i in range(vsz):
                word = Word2VecModel._readtospc(f)
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
    def _read_line_mmap(m, width, start):
        current = start+1
        while m[current:current+1] != b' ':
            current += 1
        vocab = m[start:current].decode('utf-8')
        raw = m[current+1:current+width+1]
        value = np.fromstring(raw, dtype=np.float32)
        return vocab, value, current+width + 1

    def _read_vectors_mmap(self, filename, idx, known_vocab, keep_unused):
        import mmap
        word_vectors = []
        with io.open(filename, 'rb') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                header_end = m[:50].find(b'\n')
                vsz, dsz = map(int, (m[:header_end]).split(b' '))
                width = 4 * dsz
                current = header_end + 1
                for i in range(vsz):
                    word, vec, current = Word2VecModel._read_line_mmap(m, width, current)
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

        while ch != b'\x20':
            s.extend(ch)
            ch = f.read(1)
        s = s.decode('utf-8')
        return s.strip()


@exporter
class GloVeModel(PretrainedEmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False, normalize=False, **kwargs):
        super(GloVeModel, self).__init__(filename, known_vocab, unif_weight, keep_unused, normalize, **kwargs)

    def _read_vectors(self, filename, idx, known_vocab, keep_unused, **kwargs):
        use_mmap = bool(kwargs.get('use_mmap', False))

        read_fn = self._read_vectors_mmap if use_mmap else self._read_vectors_file
        return read_fn(filename, idx, known_vocab, keep_unused)

    def _read_vectors_file(self, filename, idx, known_vocab, keep_unused):
        word_vectors = []
        with io.open(filename, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
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

    def _read_vectors_mmap(self, filename, idx, known_vocab, keep_unused):
        import mmap
        word_vectors = []
        with io.open(filename, "r", encoding="utf-8") as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                for line in iter(m.readline, ''):
                    values = line.split()
                    if len(values) == 0:
                        break
                    word = values[0]
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
        self.vocab = {}
        self.vocab["<PAD>"] = 0
        self.dsz = dsz
        self.vsz = 0

        if counts is True:
            attested = [v for v, cnt in known_vocab.items() if cnt > 0]
            for k, v in enumerate(attested):
                self.vocab[v] = k + 1
                self.vsz += 1
        else:
            print('Restoring existing vocab')
            self.vocab = known_vocab
            self.vsz = len(self.vocab) - 1

        self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv
