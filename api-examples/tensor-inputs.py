import baseline as bl
import argparse
import os
import numpy as np
"""Take a file of TSVs where format is `label<\t>content` and convert to NPZ file of vectors from pretrained embeddings

"""

BP = '../data'
TRAIN = 'stsa.binary.phrases.train'.format(BP)
VALID = 'stsa.binary.dev'
TEST = 'stsa.binary.test'
LABELS = os.path.join(BP, 'stsa.binary.labels')
W2V_GN_300 = '/data/embeddings/GoogleNews-vectors-negative300.bin'
VECTORIZERS = {'word': bl.Token1DVectorizer(mxlen=40)}


def output_file(input_file):
    return input_file + '.npz'


def convert_input(file, embeddings, batchsz=50):
    batch_x = []
    batch_y = []
    dsz = embeddings.get_dsz()
    ts = reader.load(file, vocabs={'word': embeddings.vocab}, batchsz=batchsz)
    pg = bl.create_progress_bar(len(ts))
    for batch in pg(ts):
        x = batch['word']
        B, T = x.shape
        flat_x = x.reshape(B*T)
        dense = embeddings.weights[flat_x]
        dense = dense.reshape(B, T, dsz)
        batch_x.append(dense)
        batch_y.append(batch['y'])
    return np.stack(batch_x), np.stack(batch_y)

reader = bl.TSVSeqLabelReader(VECTORIZERS,
                              clean_fn=bl.TSVSeqLabelReader.do_clean)

train_file = os.path.join(BP, TRAIN)
valid_file = os.path.join(BP, VALID)
test_file = os.path.join(BP, TEST)

# This builds a set of counters
vocabs, labels = reader.build_vocab([train_file, valid_file, test_file])
print('Writing {}'.format(LABELS))
bl.write_json(labels, LABELS)
# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = bl.PretrainedEmbeddingsModel(W2V_GN_300, known_vocab=vocabs['word'], embed_type='default', unif=0.)

print('Converting training data')
batch_x, batch_y = convert_input(train_file, embeddings, 50)
print('Saving results')
np.savez(output_file(train_file), batch_x, batch_y)

print('Converting validation data')
batch_x, batch_y = convert_input(valid_file, embeddings, 50)
print('Saving results')
np.savez(output_file(valid_file), batch_x, batch_y)

print('Converting test data')
batch_x, batch_y = convert_input(test_file, embeddings, 1)
print('Saving results')
np.savez(output_file(test_file), batch_x, batch_y)


