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
                    self.vsz = self.vsz + 1
            else:
                self.vsz = vsz

            self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))
            width = 4 * self.dsz
            k = 1
            for i in range(vsz-1):
                word = readtospc(f)
                raw = f.read(width)
                if knownvocab is not None and not word in knownvocab:
                    continue
                
                vec = np.fromstring(raw, dtype=np.float32)

                self.weights[k] = vec
                self.vocab[word] = k
                k = k + 1

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
