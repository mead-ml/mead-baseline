import tensorflow as tf
from eight_mile.utils import get_version

if not tf.executing_eagerly():
    from baseline.tf.seq2seq.decoders.v1 import *
else:
    from baseline.tf.seq2seq.decoders.v2 import *

