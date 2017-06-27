import numpy as np

def readtospc(f):

    s = bytearray()
    ch = f.read(1)

    while ch != b'\x20':
        s.extend(ch)
        ch = f.read(1)
    s = s.decode('utf-8')
    return s.strip()

class Word2VecModel:

    def __init__(self, filename, knownvocab=None, unifweight=None, keep_unused=False):

        uw = 0.0 if unifweight == None else unifweight
        self.vocab = {}
        idx = 0

        with open(filename, "rb") as f:
            header = f.readline()
            vsz, self.dsz = map(int, header.split())

            self.nullv = np.zeros(self.dsz, dtype=np.float32)
            self.vocab["<PADDING>"] = idx
            idx += 1

            word_vectors = [self.nullv]
            width = 4 * self.dsz

            for i in range(vsz):
                word = readtospc(f)
                raw = f.read(width)
                if keep_unused is False and word not in knownvocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if knownvocab and word in knownvocab:
                    knownvocab[word] = 0

                vec = np.fromstring(raw, dtype=np.float32)
                word_vectors.append(vec)

                self.vocab[word] = idx
                idx += 1

        if knownvocab is not None:
            unknown = {v: cnt for v,cnt in knownvocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        self.weights = np.array(word_vectors)
        self.vsz = self.weights.shape[0] - 1
        print(self.vsz, self.dsz)
    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

class RandomInitVecModel:

    def __init__(self, dsz, knownvocab, counts=True, unifweight=None):

        uw = 0.0 if unifweight == None else unifweight
        self.vocab = {}
        self.vocab["<PADDING>"] = 0
        self.dsz = dsz
        self.vsz = 0

        if counts is True:
            attested = {v: cnt for v,cnt in knownvocab.items() if cnt > 0}
            for k,v in enumerate(attested):
                self.vocab[v] = k
                k = k + 1
                self.vsz += 1
        else:
            print('Restoring existing vocab')
            self.vocab = knownvocab
            self.vsz = len(self.vocab) - 1

        self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

if __name__ == '__main__':
    w2v = Word2VecModel('/data/xdata/GoogleNews-vectors-negative300.bin')

    print(w2v.lookup('agjasgoikjagolkjajgr', False))
    print(w2v.lookup('agjasgoikjagolkjajgr', True))
    print(w2v.lookup('Daniel'))
