import argparse
from baseline import *

parser = argparse.ArgumentParser(description='Sequence tagger for sentences')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=str2bool, default=False)
parser.add_argument('--tensorboard', help='Turn on tensorboard reporting', type=str2bool, default=False)
parser.add_argument('--eta', default=0.01, type=float)
parser.add_argument('--embed', default=None, help='Word2Vec embeddings file')
parser.add_argument('--optim', default='sgd', help='Optim method')
parser.add_argument('--mom', default=0.9, help='SGD momentum', type=float)
parser.add_argument('--start_decay_epoch', type=int, help='At what epoch should we start decaying')
parser.add_argument('--decay_rate', default=0.0, type=float, help='Learning rate decay')
parser.add_argument('--decay_type', help='What learning rate decay schedule')
parser.add_argument('--dropout', default=0.5, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--rnntype', default='blstm', help='RNN type')
parser.add_argument('--activation', default='tanh', help='Activation type to use')
parser.add_argument('--layers', default=1, help='The depth of stacked RNNs', type=int)
parser.add_argument('--outfile', help='Output file base', default='./tagger-model')
parser.add_argument('--conll_output', default='rnn-tagger-test.txt', help='Place to put test CONLL file')
parser.add_argument('--unif', default=0.1, help='Initializer bounds for word embeddings', type=float)
parser.add_argument('--unifc', default=0.32, help='Initializer bounds for char embeddings', type=float)
parser.add_argument('--clip', default=5.0, help='Gradient clipping cutoff', type=float)
parser.add_argument('--epochs', default=40, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=10, help='Batch size', type=int)
parser.add_argument('--mxlen', default=-1, help='Max sentence length (-1 for max seen)', type=int)
parser.add_argument('--mxwlen', default=45, help='Max word length (-1 for max seen)', type=int)
parser.add_argument('--cfiltsz', help='Filter sizes', nargs='+', default=[1, 2, 3, 4, 5, 7], type=int)
parser.add_argument('--charsz', default=16, help='Char embedding depth', type=int)
parser.add_argument('--patience', default=40, help='Patience', type=int)
parser.add_argument('--hsz', default=200, help='Hidden layer size', type=int)
parser.add_argument('--wsz', default=30, help='Word embedding depth', type=int)
parser.add_argument('--nogpu', default=False, help='Use CPU (Not recommended)', type=str2bool)
parser.add_argument('--save', default='rnn-tagger', help='Save basename')
parser.add_argument('--crf', default=False, help='Use a CRF on top', type=str2bool)
parser.add_argument('--do_early_stopping', help='Should we do early stopping?', default=True, type=str2bool)
parser.add_argument('--early_stopping_metric', default='f1', help='Metric for early stopping. For IOB tagging use f1')
parser.add_argument('--web_cleanup', default=False, help='Do cleanup of web tokens?', type=str2bool)
parser.add_argument('--lower', default=True, help='Lower case word tokens?', type=str2bool)
parser.add_argument('--backend', default='tf', help='Default Deep Learning Framework')
parser.add_argument('--model_type', help='Name of model to load and train', default='default')
parser.add_argument('--trainer_type', help='Name of trainer to load and train', default='default')
parser.add_argument('--reader_type', default='default', help='reader type (defaults to CONLL)')
parser.add_argument('--proj', default=False, help='Add a hidden layer before final output', type=str2bool)
parser.add_argument('--pad_unk_test', default=False, help='Treat vocab only in test as UNK despite present embeddings')
parser.add_argument('--bounds', type=int, default=16000, help='Tell optim decay functionality how many steps before applying decay')
args = parser.parse_args()
gpu = not args.nogpu

args.reporting = setup_reporting(**vars(args))

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
    word_trans_fn = lowercase

reader = create_seq_pred_reader(args.mxlen, args.mxwlen, word_trans_fn,
                                vec_alloc, vec_shape, trim, reader_type=args.reader_type)

vocab_sources = [args.train, args.valid]

if not args.pad_unk_test:
    vocab_sources += [args.test]

vocabs = reader.build_vocab(vocab_sources)


# Vocab LUTs
embeddings = {'word': None, 'char': None}
word2index = {'word': None, 'char': None}
if args.embed:
    EmbeddingsModelType = GloVeModel if args.embed.endswith(".txt") else Word2VecModel
    embeddings['word'] = EmbeddingsModelType(args.embed, vocabs['word'], unif_weight=args.unif)
    word2index['word'] = embeddings['word'].vocab

embeddings['char'] = RandomInitVecModel(args.charsz, vocabs['char'], unif_weight=args.unifc)
word2index['char'] = embeddings['char'].vocab


ts, _ = reader.load(args.train, word2index, args.batchsz, shuffle=True)
print('Loaded training data')

vs, _ = reader.load(args.valid, word2index, args.batchsz)
print('Loaded valid data')

es, txts = reader.load(args.test, word2index, 1)
print('Loaded test data')

args.maxs = reader.max_sentence_length
args.maxw = reader.max_word_length

model = tagger.create_model(reader.label2index, embeddings, **vars(args))

tagger.fit(model, ts, vs, es, txts=txts, **vars(args))

