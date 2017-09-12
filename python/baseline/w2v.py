import numpy as np


class EmbeddingsModel(object):
    def __init__(self):
        super(EmbeddingsModel, self).__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def lookup(self, word, nullifabsent=True):
        pass


class Word2VecModel(EmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False):
        super(Word2VecModel, self).__init__()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        idx = 0

        with open(filename, "rb") as f:
            header = f.readline()
            vsz, self.dsz = map(int, header.split())

            self.nullv = np.zeros(self.dsz, dtype=np.float32)
            self.vocab["<PAD>"] = idx
            idx += 1

            word_vectors = [self.nullv]
            width = 4 * self.dsz

            for i in range(vsz):
                word = Word2VecModel._readtospc(f)
                raw = f.read(width)
                if keep_unused is False and word not in known_vocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0

                vec = np.fromstring(raw, dtype=np.float32)
                word_vectors.append(vec)

                self.vocab[word] = idx
                idx += 1

        if known_vocab is not None:
            unknown = {v: cnt for v,cnt in known_vocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        self.weights = np.array(word_vectors)
        self.vsz = self.weights.shape[0] - 1

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    @staticmethod
    def _readtospc(f):

        s = bytearray()
        ch = f.read(1)

        while ch != b'\x20':
            s.extend(ch)
            ch = f.read(1)
        s = s.decode('utf-8')
        return s.strip()

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv


class GloVeModel(EmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False):
        super(GloVeModel, self).__init__()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        idx = 1

        word_vectors = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if keep_unused is False and word not in known_vocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0
                vec = np.asarray(values[1:], dtype=np.float32)
                word_vectors.append(vec)
                self.vocab[word] = idx
                idx += 1
            self.dsz = vec.shape[0]
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
        self.vsz = self.weights.shape[0] - 1

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
