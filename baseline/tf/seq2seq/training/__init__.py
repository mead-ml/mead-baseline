from eight_mile.utils import get_version
import tensorflow as tf
if not tf.executing_eagerly():

    from baseline.tf.seq2seq.training.feed import *
else:
    from baseline.tf.seq2seq.training.eager import *
