import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
import math
from tensorflow.contrib.layers import convolution2d, max_pool2d, fully_connected, flatten, xavier_initializer
from utils import fill_y

class ConvModel:

    def save(self, sess, outdir, base):
        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(sess, basename + '.model')

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    def save_using(self, saver):
        self.saver = saver

    def restore(self, sess, indir, base):
        basename = indir + '/' + base
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'r') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: basename + '.model'})
            self.x = tf.get_default_graph().get_tensor_by_name('x:0')
            self.y = tf.get_default_graph().get_tensor_by_name('y:0')
            self.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            self.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            self.probs = tf.get_default_graph().get_tensor_by_name('output/probs:0')
        with open(basename + '.labels', 'r') as f:
            self.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            self.vocab = json.load(f)

        self.saver = tf.train.Saver(saver_def=saver_def)

    def __init__(self):
        pass

    def create_loss(self):

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.lin, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_sum(loss)

        with tf.name_scope("accuracy"):
            correct = tf.equal(self.best, tf.argmax(self.y, 1))
            all_right = tf.reduce_sum(tf.cast(correct, "float"), name="accuracy")

        return all_loss, all_right

    def inference(self, sess, examples, probs=False):
        feed_dict = {self.x: examples.x, self.pkeep: 1.0}
        if probs is True:
            return sess.run(self.probs, feed_dict=feed_dict)
        return sess.run(self.best, feed_dict=feed_dict)

    def ex2dict(self, examples, pkeep):
        return {self.x: examples.x, self.y: fill_y(len(self.labels), examples.y), self.pkeep: pkeep}

    def params(self, labels, w2v, maxlen, filtsz, cmotsz, hsz, finetune = True):
        vsz = w2v.vsz
        dsz = w2v.dsz

        self.labels = labels
        nc = len(labels)
        self.vocab = w2v.vocab
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.x = tf.placeholder(tf.int32, [None, maxlen], name="x")
        self.y = tf.placeholder(tf.int32, [None, nc], name="y")

        filtsz = [int(filt) for filt in filtsz.split(',') ]
        mxfiltsz = np.max(filtsz)
        halffiltsz = int(math.floor(mxfiltsz / 2))

        # Use pre-trained embeddings from word2vec
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name = "W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
            with tf.control_dependencies([e0]):
                # Zeropad out the word ids in the sentence to half the max
                # filter size, to make a wide convolution.  This way we
                # don't have to explicitly pad the x data upfront
                zeropad = tf.pad(self.x, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT") 
                lut = tf.nn.embedding_lookup(W, zeropad)
                expanded = tf.expand_dims(lut, -1)

        mots = []

        seed = np.random.randint(10e8)
        #init = tf.truncated_normal_initializer(stddev=0.1)
        init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        # Create parallel filter operations of different sizes
        with tf.contrib.slim.arg_scope(
                [convolution2d, fully_connected],
                weights_initializer=init,
                biases_initializer=tf.constant_initializer(0)):

            for i, fsz in enumerate(filtsz):
                with tf.name_scope('cmot-%s' % fsz) as scope:
                    conv = convolution2d(expanded, cmotsz, [fsz, dsz], [1, 1], padding='VALID', scope=scope)
                    # First dim is batch, second dim is time, third dim is feature map
                    # Max over time pooling, 2 calls below are equivalent
                    mot = tf.reduce_max(conv, [1], keep_dims=True)
                    # --------------------------
                    # siglen = maxlen - fsz + 1
                    # mot = max_pool2d(conv, [siglen, 1], 1, padding='VALID', scope=scope)
                mots.append(mot)

            combine = flatten(tf.concat(values=mots, axis=3))

            # Definitely drop out
            with tf.name_scope("dropout"):
                drop = tf.nn.dropout(combine, self.pkeep)

                # For fully connected layers, use xavier (glorot) transform
            with tf.contrib.slim.arg_scope(
                    [fully_connected],
                    weights_initializer=xavier_init):

                # This makes it more like C/W 2011
                if hsz > 0:
                    print('Adding a projection layer after MOT pooling')
                    proj = fully_connected(drop, hsz, scope='proj')
                    drop = tf.nn.dropout(proj, self.pkeep)

                with tf.name_scope("output"):
                    self.lin = fully_connected(drop, nc)
                    self.probs = tf.nn.softmax(self.lin, name="probs")
                    self.best = tf.argmax(self.lin, 1, name="best")
