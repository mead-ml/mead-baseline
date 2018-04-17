import argparse
from baseline import *

parser = argparse.ArgumentParser(description='Translate input sequence to output sequence')
parser.add_argument('--input', help='Input sequence')
parser.add_argument('--beam', default=30, help='What beam size to use', type=int)
parser.add_argument('--max_examples', default=-1, help='How many examples to show', type=int)
parser.add_argument('--nogpu', default=False, help='Dont use GPU (debug only!)', type=str2bool)
parser.add_argument('--backend', default='tf', help='Deep Learning Framework backend')
parser.add_argument('--model_type', help='Name of model to load and train', default='default')
parser.add_argument('--model', help='Name of model checkpoint to load', required=True)


def predict_sequence(text, src_vocab, dst_index2word, mxlen):
    content = text.split()
    xs = np.zeros((1, mxlen), dtype=int)
    size = min(len(content), mxlen)
    for i in range(size):
        z = src_vocab.get(content[i], 0)
        xs[0, i] = z
    lenz = np.array([len(content)])
    z = model.run({'src': xs, 'src_len': lenz})[0]
    best = z[0]
    out = []
    for i in range(len(best)):
    ##for i in range(best.shape[0]):
        word = dst_index2word.get(best[i], '<PAD>')
        if word != '<PAD>' and word != '<EOS>':
            out.append(word)
    return ' '.join(out)


args = parser.parse_args()
gpu = not args.nogpu


do_reverse = args.model_type == 'default'
if args.backend == 'pytorch':
    import baseline.pytorch.seq2seq as seq2seq
    from baseline.pytorch import *
    trim = True
else:
    from baseline.tf import *
    import baseline.tf.seq2seq as seq2seq
    trim = False

model = seq2seq.load_model(args.model, predict=True, beam=args.beam, model_type=args.model_type)
words_vocab = model.get_src_vocab()
dst_vocab = model.get_dst_vocab()
rlut = revlut(dst_vocab)


with open(args.input, "r") as f:
    for line in f:
        line = line.strip().split('\t')[0]  # In case there is a response in the same file!
        if line == '':
            print('\n')
        else:
            print(predict_sequence(line, words_vocab, rlut, 100))
