import argparse
import baseline
import sys
sys.path.append('../python/addons')
import embed_bert
from baseline.tf.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
import tensorflow as tf
import numpy as np


def get_pool_op(s):
    """Allows us to pool the features with either ``max`` or ``mean``. O.w. use identity

    :param s: The operator
    :return: The pool operation
    """
    return np.mean if s == 'mean' else np.max if s == 'max' else lambda x,axis: x

def get_vectorizer(s, vf, mxlen, lower=True):
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

    return create_vectorizer(type=vec_type, transform_fn=transform_fn, vocab_file=vf, mxlen=mxlen)

def get_embedder(embed_type, embed_file):
    """Get an embedding object by type so we can evaluate one hot vectors

    :param embed_type: (``str``) The name of the embedding in the `BASELINE_EMBEDDINGS`
    :param embed_file: (``str``) Either the file or a URL to a hub location for the model
    :return: An embeddings dict containing vocab and graph
    """
    if embed_type == 'bert':
        embed_type += '-embed'
    embed = baseline.load_embeddings('word', embed_type=embed_type,
                                     embed_file=embed_file, keep_unused=True, trainable=False)
    return embed

parser = argparse.ArgumentParser(description='Encode a sentence as an embedding')
parser.add_argument('--embed_file', help='embedding file')
parser.add_argument('--type', default='default', choices=['bert', 'default'])
parser.add_argument('--sentences', required=True)
parser.add_argument('--output', default='embeddings.npz')
parser.add_argument('--pool', default=None)
parser.add_argument('--lower', type=baseline.str2bool, default=True)
parser.add_argument('--vocab_file')
parser.add_argument('--max_length', type=int, default=100)
args = parser.parse_args()


# Create our vectorizer according to CL
vectorizer = get_vectorizer(args.type, args.vocab_file, args.max_length)

# Pool operation for once we have np.array
pooling = get_pool_op(args.pool)

# Make a session
with tf.Session() as sess:
    # Get embeddings
    embed = get_embedder(args.type, args.embed_file)

    # This is our decoder graph object
    embedder = embed['embeddings']

    # This is the vocab
    vocab = embed['vocab']

    # Declare a tf graph operation
    y = embedder.encode()
    sess.run(tf.global_variables_initializer())

    # Read a newline separated file of sentences
    with open(args.sentences) as f:

        vecs = []
        for line in f:
            # For each sentence
            tokens = line.strip().split()
            # Run vectorizer to get ints and length of vector
            one_hot, sentence_len = vectorizer.run(tokens, vocab)
            # Expand so we have a batch dim
            one_hot_batch = np.expand_dims(one_hot, 0)
            # Evaluate the graph and get rid of batch dim
            sentence_emb = sess.run(y, feed_dict={embedder.x: one_hot_batch}).squeeze()
            # Run the pooling operator
            sentence_vec = pooling(sentence_emb, axis=0)
            #print(sentence_vec)
            vecs.append(sentence_vec)
        # Store to file
        np.savez(args.output, np.stack(vecs))
