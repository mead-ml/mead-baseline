from baseline import *
import argparse

def num_steps_per_epoch(num_examples, nbptt, batchsz):
    rest = num_examples // batchsz
    return rest // nbptt


parser = argparse.ArgumentParser(description='Language Modeler')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=str2bool, default=False)
parser.add_argument('--tensorboard', help='Turn on tensorboard reporting', type=str2bool, default=False)
parser.add_argument('--eta', default=1, help='Initial learning rate', type=float)
parser.add_argument('--embed', default=None, help='Word2Vec embeddings file')
parser.add_argument('--optim', default='sgd', help='Optim method')
parser.add_argument('--decay', default=0, help='LR decay', type=float)
parser.add_argument('--mom', default=0, help='SGD momentum', type=float)
parser.add_argument('--dropout', default=0.5, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--rnntype', default='lstm', help='RNN type')
parser.add_argument('--layers', default=2, help='The depth of stacked RNNs', type=int)
parser.add_argument('--outfile', help='Output file base', default='./lang-model')
parser.add_argument('--unif', default=0.1, help='Initializer bounds for embeddings', type=float)
parser.add_argument('--clip', default=5.0, help='Gradient clipping cutoff', type=float)
parser.add_argument('--epochs', default=30, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=20, help='Batch size', type=int)
parser.add_argument('--nbptt', default=35, help='Steps of backprop through time', type=int)
parser.add_argument('--mxwlen', default=40, help='Max word length', type=int)
parser.add_argument('--filtsz', help='Filter sizes', nargs='+', default=[1, 2, 3, 4, 5, 7], type=int)
parser.add_argument('--charsz', default=16, help='Char embedding depth', type=int)
parser.add_argument('--patience', default=70, help='Patience', type=int)
parser.add_argument('--hsz', default=100, help='Hidden layer size', type=int)
parser.add_argument('--wsz', default=30, help='Word embedding depth', type=int)
parser.add_argument('--valsplit', default=0.15, help='Validation split if no valid set', type=float)
parser.add_argument('--nogpu', default=False, help='Use CPU (Not recommended)', type=str2bool)
parser.add_argument('--model_file', default='wchar_lm.model', help='Save basename')
parser.add_argument('--test_thresh', default=10, help='How many epochs improvement required before testing', type=int)
parser.add_argument('--model_type', default='default', help='What type of language model')
parser.add_argument('--do_early_stopping', help='Should we do early stopping?', default=True, type=str2bool)
parser.add_argument('--early_stopping_metric', default='avg_loss', help='Metric to use for early stopping')
parser.add_argument('--start_decay_epoch', default=6, type=int, help='At what epoch should we start decaying')
parser.add_argument('--decay_rate', default=1.2, type=float, help='Learning rate decay')
parser.add_argument('--decay_type', default='zaremba', help='What learning rate decay schedule')
parser.add_argument('--backend', default='tf', help='Default Deep Learning Framework')
args = parser.parse_args()


args.reporting = setup_reporting(**vars(args))

if args.backend == 'tf':
    import baseline.tf.lm as lm

reader = create_lm_reader(args.mxwlen, args.nbptt)
vocab_ch, vocab_word, num_words = reader.build_vocab([args.train, args.valid, args.test])


# Vocab LUTs
word_vocab = None
char_vocab = None

# No matter what we will create a vocab for words, since that is what we are emitting
word_vec = None
if args.embed:
    word_vec = w2v.Word2VecModel(args.embed, vocab_word, args.unif)
    word_vocab = word_vec.vocab
# TODO: Fix this to be a boolean for use word vectors
else:
    print('Creating new embedding weights')
    word_vec = w2v.RandomInitVecModel(args.hsz, vocab_word, unif_weight=args.unif)
    word_vocab = word_vec.vocab

char_vec = w2v.RandomInitVecModel(args.charsz, vocab_ch, unif_weight=args.unif)
char_vocab = char_vec.vocab

ts = reader.load(args.train, word_vocab, char_vocab, num_words[0], batchsz=args.batchsz)
print('Loaded training data')

vs = reader.load(args.valid, word_vocab, char_vocab, num_words[1], batchsz=args.batchsz)
print('Loaded validation data')

es = reader.load(args.test, word_vocab, char_vocab, num_words[2], batchsz=args.batchsz)

print('Using %d examples for training' % num_words[0])
print('Using %d examples for validation' % num_words[1])
print('Using %d examples for test' % num_words[2])
args.maxw = reader.max_word_length
model = lm.create_model(word_vec, char_vec, **vars(args))
steps_per_epoch = num_steps_per_epoch(num_words[0], args.nbptt, args.batchsz)
first_range = int(args.start_decay_epoch * steps_per_epoch)
args.bounds = [first_range] + list(np.arange(args.start_decay_epoch + 1, args.epochs + 1, dtype=np.int32) * steps_per_epoch)

lm.fit(model, ts, vs, es, **vars(args))

