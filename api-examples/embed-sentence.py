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
    return np.mean if s == 'mean' else np.max if s == 'max' else lambda x,axis: x

def get_vectorizer(s, vf, mxlen, lower=True):
    vec_type = 'token1d'
    transform_fn = baseline.lowercase if lower else None
    if s == 'bert':
        vec_type = 'wordpiece1d'

    return create_vectorizer(type=vec_type, transform_fn=baseline.lowercase, vocab_file=vf, mxlen=mxlen)

def get_embedder(embed_type, embed_file):
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


vectorizer = get_vectorizer(args.type, args.vocab_file, args.max_length)
pooling = get_pool_op(args.pool)

with tf.Session() as sess:
    # Declare the decoder graph
    embed = get_embedder(args.type, args.embed_file)
    embedder = embed['embeddings']
    vocab = embed['vocab']
    y = embedder.encode()
    sess.run(tf.global_variables_initializer())

    with open(args.sentences) as f:

        vecs = []
        for line in f:
            # For each sentence embedding
            tokens = line.strip().split()
            one_hot, sentence_len = vectorizer.run(tokens, vocab)
            one_hot_batch = np.expand_dims(one_hot, 0)
            sentence_emb = sess.run(y, feed_dict={embedder.x: one_hot_batch}).squeeze()
            sentence_vec = pooling(sentence_emb, axis=0)
            #print(sentence_vec)
            vecs.append(sentence_vec)
        np.savez(args.output, np.stack(vecs))
