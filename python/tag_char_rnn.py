import argparse
from baseline import *

parser = argparse.ArgumentParser(description='Sequence tagger for sentences')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=bool, default=False)
parser.add_argument('--eta', default=0.01, type=float)
parser.add_argument('--embed', default=None, help='Word2Vec embeddings file')
parser.add_argument('--optim', default='adadelta', help='Optim method')
parser.add_argument('--decay', default=0, help='LR decay', type=float)
parser.add_argument('--mom', default=0.9, help='SGD momentum', type=float)
parser.add_argument('--dropout', default=0.5, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--rnntype', default='blstm', help='RNN type')
parser.add_argument('--layers', default=1, help='The depth of stacked RNNs', type=int)
parser.add_argument('--outfile', help='Output file base', default='./tagger-model')
parser.add_argument('--conll_output', default='rnn-tagger-test.txt', help='Place to put test CONLL file')
parser.add_argument('--unif', default=0.1, help='Initializer bounds for embeddings', type=float)
parser.add_argument('--clip', default=5.0, help='Gradient clipping cutoff', type=float)
parser.add_argument('--epochs', default=400, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=20, help='Batch size', type=int)
parser.add_argument('--mxlen', default=-1, help='Max sentence length', type=int)
parser.add_argument('--mxwlen', default=40, help='Max word length', type=int)
parser.add_argument('--cfiltsz', help='Filter sizes', nargs='+', default=[1,2,3,4,5,7], type=int)
parser.add_argument('--charsz', default=16, help='Char embedding depth', type=int)
parser.add_argument('--patience', default=20, help='Patience', type=int)
parser.add_argument('--hsz', default=200, help='Hidden layer size', type=int)
parser.add_argument('--wsz', default=30, help='Word embedding depth', type=int)
parser.add_argument('--valsplit', default=0.15, help='Validation split if no valid set', type=float)
parser.add_argument('--nogpu', default=False, help='Use CPU (Not recommended)', type=bool)
parser.add_argument('--save', default='rnn-tagger', help='Save basename')
parser.add_argument('--test_thresh', default=10, help='How many epochs improvement required before testing', type=int)
parser.add_argument('--crf', default=False, help='Use a CRF on top', type=bool)
parser.add_argument('--early_stopping_metric', default='f1', help='Metric for early stopping. For IOB tagging use f1')
parser.add_argument('--web_cleanup', default=False, help='Do cleanup of web tokens?', type=bool)
parser.add_argument('--lower', default=False, help='Lower case word tokens?', type=bool)
parser.add_argument('--backend', default='tf', help='Default Deep Learning Framework')
args = parser.parse_args()
gpu = not args.nogpu


args.reporting = setup_reporting(args.visdom)

if args.backend == 'pytorch':
    from baseline.pytorch import long_0_tensor_alloc as vec_alloc
    from baseline.pytorch import tensor_shape as vec_shape
    import baseline.pytorch.tagger as tagger
    trim = True
else:
    from numpy import zeros as vec_alloc
    from numpy import shape as vec_shape
    if args.backend == 'tf':
        import baseline.tf.tagger as tagger
        trim = False

word_trans_fn = None
if args.web_cleanup is True:
    print('Web-ish data cleanup')
    word_trans_fn = CONLLSeqReader.web_cleanup
elif args.lower is True:
    print('Lower-case word tokens')
    word_trans_fn = str.lower

reader = CONLLSeqReader(args.mxlen, args.mxwlen, word_trans_fn=word_trans_fn,
                        vec_alloc=vec_alloc, vec_shape=vec_shape, trim=trim)
vocab_ch, vocab_word = reader.build_vocab([args.train, args.test, args.valid])


# Vocab LUTs
word_vocab = None
char_vocab = None

word_vec = None
if args.embed:
    word_vec = GloVeModel(args.embed, vocab_word, unif_weight=args.unif) if args.embed.endswith(".txt") else Word2VecModel(args.embed, vocab_word, unif_weight=args.unif)
    word_vocab = word_vec.vocab

char_vec = RandomInitVecModel(args.charsz, vocab_ch, unif_weight=args.unif)
char_vocab = char_vec.vocab
print(char_vocab)

ts, _ = reader.load(args.train, word_vocab, char_vocab, args.batchsz, shuffle=True)
print('Loaded training data')

vs, _ = reader.load(args.valid, word_vocab, char_vocab, args.batchsz)
print('Loaded valid data')

es, txts = reader.load(args.test, word_vocab, char_vocab, 2)
print('Loaded test data')

args.maxs = reader.max_sentence_length
args.maxw = reader.max_word_length

model = tagger.create_model(reader.label2index, word_vec, char_vec, **vars(args))

tagger.fit(model, ts, vs, es, **vars(args))

