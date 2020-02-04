"""Program to generate preprocessed N-gram vocabulary for contextual embeddings

To do this, we need to collect all of the N-gram vocab from a corpus, and then
run each N-gram chunk through the embedder.  The vocabulary for the embedder is
going to be 1-grams and it will process an array.  For example, if we have a sample
sentence: "The dog crossed the road", and we are collecting trigrams, our vocab will
include entries like: `["<PAD> The dog", "The dog crossed", "dog crossed the",...]` etc.

To prepare the output file (which will be in `word2vec` binary format, we evaluate N-gram
and we populate a single word2vec entry with delimiters:

`"The@@dog@@crossed <vector>`

The `<vector>` is found by running, e.g. ELMo with the entry "The dog crossed", which yields
a `Time x Word` output, and we select a pooled representation and send it back

"""
import argparse
import baseline
import sys
sys.path.append('../python/addons')
import embed_elmo
import embed_bert
from baseline.tf.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import create_vectorizer, TextNGramVectorizer
from baseline.reader import CONLLSeqReader, TSVSeqLabelReader
from baseline.embeddings import write_word2vec_file
from baseline.utils import ngrams
import tensorflow as tf
import numpy as np
import codecs
import re
from collections import Counter
BATCHSZ = 32


def pool_op(embedding, name):
    if len(embedding.shape) == 3:
        assert embedding.shape[0] == 1
        embedding = embedding.squeeze()
    T = embedding.shape[0]
    if T == 1:
        return embedding.squeeze()
    if name == 'mean':
        return np.mean(embedding, axis=0)
    if name == 'sum':
        return np.sum(embedding, axis=0)
    if name == 'max':
        return np.max(embedding, axis=0)
    if name == 'last':
        return embedding[-1]
    elif name == 'first':
        embedding[0]
    center = T//2 + 1
    return embedding[center]


def get_unigram_vectorizer(s, vf, mxlen, lower=True):
    """Get a vectorizer object by name from `BASELINE_VECTORIZERS` registry

    :param s: The name of the vectorizer
    :param vf: A vocabulary file (which might be ``None``)
    :param mxlen: The vector length to use
    :param lower: (``bool``) should we lower case?  Defaults to ``True``
    :return: A ``baseline.Vectorizer`` subclass
    """
    vec_type = 'token1d'
    transform_fn = baseline.lowercase if lower else None
    if s == 'bert':
        vec_type = 'wordpiece1d'
    if s == 'elmo':
        vec_type = 'elmo'
    return create_vectorizer(type=vec_type, transform_fn=transform_fn, vocab_file=vf, mxlen=mxlen)


def get_embedder(embed_type, embed_file):
    """Get an embedding object by type so we can evaluate one hot vectors

    :param embed_type: (``str``) The name of the embedding in the `BASELINE_EMBEDDINGS`
    :param embed_file: (``str``) Either the file or a URL to a hub location for the model
    :return: An embeddings dict containing vocab and graph
    """
    if embed_type == 'bert' or embed_type == 'elmo':
        embed_type += '-embed'
    embed = baseline.load_embeddings('word', embed_type=embed_type,
                                     embed_file=embed_file, keep_unused=True, trainable=False, known_vocab={})
    return embed

parser = argparse.ArgumentParser(description='Encode a sentence as an embedding')
parser.add_argument('--input_embed', help='Input embedding model. This will typically be a serialized contextual model')
parser.add_argument('--type', default='default', choices=['elmo', 'default'])
parser.add_argument('--files', required=True, nargs='+')
parser.add_argument('--output_embed', default='elmo.bin', help='Output embedding model in word2vec binary format')
parser.add_argument('--lower', type=baseline.str2bool, default=False)
parser.add_argument('--vocab_file', required=False, help='Vocab file (required only for BERT)')
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--ngrams', type=int, default=3)
parser.add_argument('--reader', default='conll', help='Supports CONLL or TSV')
parser.add_argument('--column', default='0', help='Default column to read features from')
parser.add_argument('--op', type=str, default='mean')
args = parser.parse_args()


# Create our vectorizer according to CL
uni_vec = get_unigram_vectorizer(args.type, args.vocab_file, args.ngrams)

def read_tsv_features(files, column, filtsz, lower):
    """Read features from CONLL file, yield a Counter of words

    :param files: Which files to read to form the vocab
    :param column: What column to read from (defaults to '0')
    :param filtsz: An integer value for the ngram length, e.g. 3 for trigram
    :param lower: Use lower case
    :return: A Counter of words
    """

    words = Counter()
    text_column = int(column)

    transform_fn = lambda z: z.lower() if lower else z

    for file_name in files:
        if file_name is None:
            continue
        with codecs.open(file_name, encoding='utf-8', mode='r') as f:
            for il, line in enumerate(f):
                columns = line.split('\t')
                text = columns[text_column]
                sentence = text.split()
                if len(text) == 0:
                    print('Warning, invalid text at {}'.format(il))
                    continue
                pad = ['<UNK>'] * (filtsz//2)
                words.update(ngrams(pad + [transform_fn(x) for x in sentence] + pad, filtsz=filtsz))
    return words


def read_conll_features(files, column, filtsz, lower):
    """Read features from CONLL file, yield a Counter of words

    :param files: Which files to read to form the vocab
    :param column: What column to read from (defaults to '0')
    :param filtsz: An integer value for the ngram length, e.g. 3 for trigram
    :param lower: Use lower-case
    :return: A Counter of words
    """
    words = Counter()
    conll = CONLLSeqReader(None)

    text_column = str(column)
    # This tabulates all of the ngrams
    for file in files:
        print('Adding vocab from {}'.format(file))

        examples = conll.read_examples(file)

        transform_fn = lambda z: z.lower() if lower else z
        for sentence in examples:
            pad = ['<UNK>'] * (filtsz//2)
            words.update(ngrams(pad + [transform_fn(x[text_column]) for x in sentence] + pad, filtsz=filtsz))
    return words

reader_fn = read_conll_features if args.reader == 'conll' else read_tsv_features
words = reader_fn(args.files, args.column, args.ngrams, args.lower)
# print them too
print(words.most_common(25))

# build a vocab for the output file comprised of the ngram words
output_vocab = list(words)


# Make a session
with tf.compat.v1.Session() as sess:
    # Get embeddings
    embed = get_embedder(args.type, args.input_embed)

    # This is our decoder graph object
    embedder = embed['embeddings']

    # This is the vocab
    vocab = embed['vocab']

    # Declare a tf graph operation
    y = embedder.encode()

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    vecs = []
    one_hot_batch = []
    for i, token in enumerate(output_vocab):
        tokens = token.split('@@')
        # Run vectorizer to get ints and length of vector
        if i % BATCHSZ == 0 and i > 0:
            # This resets the session, which is needed for ELMo to get same results when batching
            sess.run(init_op)
            vecs += [pool_op(emb, args.op) for emb in sess.run(y, feed_dict={embedder.x: one_hot_batch})]
            one_hot_batch = []
        one_hot, sentence_len = uni_vec.run(tokens, vocab)
        one_hot_batch.append(one_hot)
    if one_hot_batch:
        sess.run(init_op)
        vecs += [pool_op(emb, args.op) for emb in sess.run(y, feed_dict={embedder.x: one_hot_batch})]

    write_word2vec_file(args.output_embed, output_vocab, vecs)
