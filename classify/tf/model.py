import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from utils import fill_y
import json
import math
class ConvModel:

    def save(self, sess, outdir, base):
        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w+b') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(sess, basename + '.model')

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)


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

    def createLoss(self):

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(self.lin, tf.cast(self.y, "float"))
            all_loss = tf.reduce_sum(loss)


        with tf.name_scope("accuracy"):
            correct = tf.equal(self.best, tf.argmax(self.y, 1))
            all_right = tf.reduce_sum(tf.cast(correct, "float"), name="accuracy")

        return all_loss, all_right

    def inference(self, sess, batch, probs=False):
        feed_dict = {self.x: batch["x"], self.pkeep: 1.0}
        if probs is True:
            return sess.run(self.probs, feed_dict=feed_dict)
        return sess.run(self.best, feed_dict=feed_dict)

    def ex2dict(self, example, pkeep):
        return {self.x: example["x"], self.y: fill_y(len(self.labels), example["y"]), self.pkeep: pkeep}

    def params(self, labels, w2v, maxlen, filtsz, cmotsz):
        vsz = w2v.vsz
        dsz = w2v.dsz

        self.labels = labels
        nc = len(labels)
        self.vocab = w2v.vocab
        expanded = self.input2expanded(labels, w2v, maxlen)
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        

        filtsz = [int(filt) for filt in filtsz.split(',') ]

        mots = []
        for i, fsz in enumerate(filtsz):
            with tf.name_scope('cmot-%s' % fsz):
                siglen = maxlen - fsz + 1
                W = tf.Variable(tf.truncated_normal([fsz, dsz, 1, cmotsz],
                                                    stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[cmotsz], dtype=tf.float32), name="b")
                conv = tf.nn.conv2d(expanded, 
                                    W, strides=[1,1,1,1], 
                                    padding="VALID", name="conv")
                
                activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

                mot = tf.nn.max_pool(activation,
                                     ksize=[1, siglen, 1, 1],
                                     strides=[1,1,1,1],
                                     padding="VALID",
                                     name="pool")
                mots.append(mot)
            
        cmotsz_all = cmotsz * len(mots)
        combine = tf.reshape(tf.concat(3, mots), [-1, cmotsz_all])
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(combine, self.pkeep)

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([cmotsz_all, nc],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")
            self.lin = tf.matmul(drop, W) + b
            self.probs = tf.nn.softmax(self.lin, name="probs")
            self.best = tf.argmax(self.lin, 1, name="best")

    def input2expanded(self, labels, w2v, maxlen):
        pass

class ConvModelStatic(ConvModel):

    def input2expanded(self, labels, w2v, maxlen):
        vsz = w2v.vsz
        dsz = w2v.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.float32, [None, maxlen, dsz], name="x")
        self.y = tf.placeholder(tf.int32, [None, nc], name="y")
        
        with tf.name_scope('expand'):
            expanded = tf.expand_dims(self.x, -1)

        return expanded

class ConvModelFineTune(ConvModel):

    def input2expanded(self, labels, w2v, maxlen):

        vsz = w2v.vsz
        dsz = w2v.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.int32, [None, maxlen], name="x")
        self.y = tf.placeholder(tf.int32, [None, nc], name="y")
        
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name = "W")

            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
        
            with tf.control_dependencies([e0]):
                lut = tf.nn.embedding_lookup(W, self.x)
                expanded = tf.expand_dims(lut, -1)

        return expanded


