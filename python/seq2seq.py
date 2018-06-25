import argparse
from baseline import *

parser = argparse.ArgumentParser(description='Sequence to sequence learning')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=str2bool, default=False)
parser.add_argument('--tensorboard', help='Turn on tensorboard reporting', type=str2bool, default=False)
parser.add_argument('--eta', default=0.001, help='Initial learning rate.', type=float)
parser.add_argument('--mom', default=0.9, help='Momentum (if SGD)', type=float)
parser.add_argument('--embed1', help='Word2Vec embeddings file (1)')
parser.add_argument('--embed2', help='Word2Vec embeddings file (2)')
parser.add_argument('--rnntype', default='lstm', help='(lstm|gru)')
parser.add_argument('--optim', default='adam', help='Optim method')
parser.add_argument('--dropout', default=0.5, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file')
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file')
parser.add_argument('--unif', default=0.25, help='Initializer bounds for embeddings', type=float)
parser.add_argument('--epochs', default=60, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=50, help='Batch size', type=int)
parser.add_argument('--mxlen', default=1000, help='Max length', type=int)
parser.add_argument('--patience', default=10, help='Patience', type=int)
parser.add_argument('--hsz', default=100, help='Hidden layer size', type=int)
parser.add_argument('--dsz', default=300, help='Embeddings size if pre-trained word vectors are not given', type=int)
parser.add_argument('--outfile', help='Output file base', default='./seq2seq-model')
parser.add_argument('--clip', default=1.0, help='Gradient clipping', type=float)
parser.add_argument('--layers', default=1, help='Number of LSTM layers for encoder/decoder', type=int)
parser.add_argument('--sharedv', default=False, help='Share vocab between source and destination', type=str2bool)
parser.add_argument('--showex', default=True, help='Show generated examples every few epochs', type=str2bool)
parser.add_argument('--sample', default=False, help='If showing examples, sample?', type=str2bool)
parser.add_argument('--topk', default=5, help='If sampling in examples, prunes to topk', type=int)
parser.add_argument('--max_examples', default=5, help='How many examples to show', type=int)
parser.add_argument('--nogpu', default=False, help='Dont use GPU (debug only!)', type=str2bool)
parser.add_argument('--pdrop', default=0.5, help='Dropout', type=float)
parser.add_argument('--backend', default='tf', help='Deep Learning Framework backend')
parser.add_argument('--pair_suffix', default=None, nargs='+', help='list of suffixes to give if parallel corpora')
parser.add_argument('--do_early_stopping', help='Should we do early stopping?', default=True, type=str2bool)
parser.add_argument('--early_stopping_metric', default='avg_loss', help='Metric for early stopping')
parser.add_argument('--vocab', default=None, help='vocab (basename) file to give if it exists')
parser.add_argument('--reader_type', default='default', help='reader type')
parser.add_argument('--model_type', help='Name of model to load and train', default='default')
parser.add_argument('--trainer_type', help='Name of trainer to load and train', default='default')
parser.add_argument('--arc_state', help='Create arc between encoder final state and decoder init state', default=False, type=str2bool)
parser.add_argument('--gpus', help='GPUs', nargs='+', default=[], type=int)
args = parser.parse_args()
gpu = not args.nogpu

args.reporting = setup_reporting(**vars(args))

do_reverse = args.model_type == 'default'
if args.backend == 'pytorch':
    import baseline.pytorch.seq2seq as seq2seq
    from baseline.pytorch import *
    show_ex_fn = show_examples_pytorch
    alloc_fn = long_0_tensor_alloc
    src_vec_trans = tensor_reverse_2nd if do_reverse else None
    trim = True
else:
    from baseline.tf import *
    import baseline.tf.seq2seq as seq2seq
    from numpy import zeros as alloc_fn
    show_ex_fn = show_examples_tf
    src_vec_trans = reverse_2nd if do_reverse else None
    trim = False

reader = create_parallel_corpus_reader(args.mxlen, alloc_fn, trim, src_vec_trans,
                                       pair_suffix=args.pair_suffix, reader_type=args.reader_type)
if args.vocab is not None:
    vocab_list = [args.vocab]
else:
    vocab_list = [args.train, args.valid, args.test]

vocab1, vocab2 = reader.build_vocabs(vocab_list)

if args.embed1:
    EmbeddingsModelType = GloVeModel if args.embed1.endswith(".txt") else Word2VecModel
    embed1 = EmbeddingsModelType(args.embed1, vocab1, unif_weight=args.unif)
else:
    embed1 = RandomInitVecModel(args.dsz, vocab1, unif_weight=args.unif)

if args.embed2:
    EmbeddingsModelType = GloVeModel if args.embed2.endswith(".txt") else Word2VecModel
    embed2 = EmbeddingsModelType(args.embed2, vocab2, unif_weight=args.unif)
else:
    embed2 = RandomInitVecModel(args.dsz, vocab2, unif_weight=args.unif)

ts = reader.load(args.train, embed1.vocab, embed2.vocab, args.batchsz, shuffle=True)
es = reader.load(args.test, embed1.vocab, embed2.vocab, args.batchsz)
print('Finished loading datasets')
rlut1 = revlut(embed1.vocab)
rlut2 = revlut(embed2.vocab)

model = seq2seq.create_model(embed1, embed2, **vars(args))

# This code is framework specific
if args.showex:
    args.after_train_fn = lambda model: show_ex_fn(model, es, rlut1, rlut2, embed2,
                                                   args.mxlen, args.sample, args.topk, args.max_examples, reverse=do_reverse)

seq2seq.fit(model, ts, es, **vars(args))
