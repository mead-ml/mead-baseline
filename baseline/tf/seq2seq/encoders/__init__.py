import tensorflow as tf
from eight_mile.utils import get_version

if not tf.executing_eagerly():
    from baseline.tf.seq2seq.encoders.v1 import *
else:
    from baseline.tf.seq2seq.encoders.v2 import *
