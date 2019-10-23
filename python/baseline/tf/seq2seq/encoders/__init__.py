import tensorflow as tf
from eight_mile.utils import get_version

if get_version(tf) < 2:
    from baseline.tf.seq2seq.encoders.v1 import *
else:
    from baseline.tf.seq2seq.encoders.v2 import *
