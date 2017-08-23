import numpy as np
import argparse
from baseline import *
from os import sys, path, makedirs

parser = argparse.ArgumentParser(description='Train a text classifier')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=bool, default=False)
parser.add_argument('--tensorboard', help='Turn on tensorboard reporting', type=bool, default=False)
parser.add_argument('--eta', help='Initial learning rate', default=0.01, type=float)
parser.add_argument('--mom', help='SGD Momentum', default=0.9, type=float)
parser.add_argument('--embed', help='Word2Vec embeddings file', required=True)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--save', help='Save basename', default='classify_sentence_pytorch')
parser.add_argument('--nogpu', help='Do not use GPU', default=False)
parser.add_argument('--optim', help='Optim method', default='adam', choices=['adam', 'adagrad', 'adadelta', 'sgd'])
parser.add_argument('--dropout', help='Dropout probability', default=0.5, type=float)
parser.add_argument('--unif', help='Initializer bounds for embeddings', default=0.25)
parser.add_argument('--epochs', help='Number of epochs', default=25, type=int)
parser.add_argument('--batchsz', help='Batch size', default=50, type=int)
parser.add_argument('--mxlen', help='Max length', default=100, type=int)
parser.add_argument('--patience', help='Patience', default=10, type=int)
parser.add_argument('--cmotsz', help='Hidden layer size', default=100, type=int)
parser.add_argument('--filtsz', help='Filter sizes', nargs='+', default=[3,4,5], type=int)
parser.add_argument('--clean', help='Do cleaning', action='store_true', default=True)
parser.add_argument('--static', help='Fix pre-trained embeddings weights', action='store_true')
parser.add_argument('--valsplit', help='Validation split if no valid set', default=0.15, type=float)
parser.add_argument('--outfile', help='Output file base', default='./classify-model')
parser.add_argument('--backend', help='Which deep learning framework to use', default='tf')
parser.add_argument('--keep_unused', help='Keep unused vocabulary terms as word vectors', default=False)
parser.add_argument('--do_early_stopping', help='Should we do early stopping?', default=True, type=bool)
parser.add_argument('--early_stopping_metric', help='What metric should we use if stopping early', default='acc')
parser.add_argument('--model_type', help='Name of model to load and train', default='default')
parser.add_argument('--rev', help='Time reverse input text', default=False, type=bool)
args = parser.parse_args()


if args.backend == 'pytorch':
    from baseline.pytorch import long_0_tensor_alloc as vec_alloc
    from baseline.pytorch import tensor_reverse_2nd as rev2nd
    import baseline.pytorch.classify as classify
    zeropadding = np.max(args.filtsz)
else:
    # Everything else uses numpy
    from numpy import zeros as vec_alloc
    from baseline.data import reverse_2nd as rev2nd
    if args.backend == 'keras':
        zeropadding = np.max(args.filtsz)

        import baseline.keras.classify as classify
    else:
        import baseline.tf.classify as classify
        # For tensorflow, use tf.pad internally in the model
        zeropadding = 0

args.reporting = setup_reporting(**vars(args))

clean_fn = TSVSeqLabelReader.do_clean if args.clean else None
src_vec_trans = rev2nd if args.rev else None

print(clean_fn, src_vec_trans)
reader = create_pred_reader(args.mxlen, zeropadding, clean_fn, vec_alloc, src_vec_trans)
vocab, labels = reader.build_vocab([args.train, args.test, args.valid])
unif = 0 if args.static else args.unif
embeddings = GloVeModel(args.embed,
                        vocab,
                        unif_weight=args.unif) if args.embed.endswith(".txt") else Word2VecModel(args.embed,
                                                                                                 vocab,
                                                                                                 unif,
                                                                                                 keep_unused=args.keep_unused)


ts = reader.load(args.train, embeddings.vocab, args.batchsz, shuffle=True)
print('Loaded training data')

vs = reader.load(args.valid, embeddings.vocab, args.batchsz)
print('Loaded valid data')

es = reader.load(args.test, embeddings.vocab, 2)
print('Loaded test data')
print('Number of labels found: [%d]' % len(labels))

model = classify.create_model(embeddings, labels,
                              model_type=args.model_type,
                              mxlen=args.mxlen,
                              unif=args.unif,
                              filtsz=args.filtsz,
                              cmotsz=args.cmotsz,
                              dropout=args.dropout,
                              finetune=not args.static)

classify.fit(model, ts, vs, es, **vars(args))
