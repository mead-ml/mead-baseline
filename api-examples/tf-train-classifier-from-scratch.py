import baseline as bl
import baseline.tf.classify as classify
import os

BP = '../data'
TRAIN = 'stsa.binary.phrases.train'.format(BP)
VALID = 'stsa.binary.dev'
TEST = 'stsa.binary.test'
W2V_GN_300 = '/data/embeddings/GoogleNews-vectors-negative300.bin'
# The `vectorizer`'s job is to take in a set of tokens and turn them into a numpy array
feature_desc = {
    'word': {
        'vectorizer': bl.Token1DVectorizer(mxlen=40),
        'embed': { 'file': W2V_GN_300, 'type': 'default', 'unif': 0.25 }
    }
}
# Create a reader that is using our vectorizers to parse a TSV file
# with rows like:
# <label>\t<sentence>\n

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = bl.TSVSeqLabelReader(vectorizers,
                              clean_fn=bl.TSVSeqLabelReader.do_clean)

train_file = os.path.join(BP, TRAIN)
valid_file = os.path.join(BP, VALID)
test_file = os.path.join(BP, TEST)

# This builds a set of counters
vocabs, labels = reader.build_vocab([train_file,
                                     valid_file,
                                     test_file])

# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings[k] = bl.load_embeddings(embed_config['file'],
                                       known_vocab=vocabs[k],
                                       embed_type=embed_config.get('type',
                                                                   'default'),
                                       unif=embed_config.get('unif', 0.),
                                       use_mmap=True)

    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings[k].vocab

# Now create the model we want to train.  There are lots of HPs we can pass in
# but for this simple example, use basic defaults
model = classify.create_model(embeddings, labels, cmotsz=200, filtsz=[3, 4, 5])

ts = reader.load(train_file, vocabs=vocabs, batchsz=50)
vs = reader.load(valid_file, vocabs=vocabs, batchsz=50)
es = reader.load(test_file, vocabs=vocabs, batchsz=1)

classify.fit(model, ts, vs, es, epochs=2, otim='adadelta')
