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

    def __init__(self, filename, knownvocab=None, unifweight=None):

        uw = 0.0 if unifweight == None else unifweight
        self.vocab = {}
        self.vocab["<PADDING>"] = 0
        with open(filename, "rb") as f:
            header = f.readline()
            vsz, self.dsz = map(int, header.split())

            if knownvocab is not None:
                self.vsz = 0
                for v in knownvocab:
                    self.vsz += 1
            else:
                self.vsz = vsz

            self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))
            width = 4 * self.dsz
            k = 1
            # All attested word vectors
            for i in range(vsz):
                word = readtospc(f)
                raw = f.read(width)
                # If vocab list, not in: dont add, in:add, drop from list
                if word in self.vocab:
                    continue

                if knownvocab is not None:
                    if word not in knownvocab:
                        continue

                    # Otherwise drop freq to 0, for later
                    knownvocab[word] = 0
                vec = np.fromstring(raw, dtype=np.float32)
                self.weights[k] = vec
                self.vocab[word] = k
                k = k + 1

            # Anything left over, unattested in w2v model, just use a random
            # initialization
        if knownvocab is not None:
            unknown = {v: cnt for v,cnt in knownvocab.items() if cnt > 0}
            for v in unknown:
                self.vocab[v] = k
                k = k + 1

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

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
            for k,v in enumerate(knownvocab):
                self.vocab[k] = v

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
