import tensorflow as tf
from eight_mile.utils import get_version

if get_version(tf) < 2:
    from baseline.tf.seq2seq.decoders.v1 import *
else:
    from baseline.tf.seq2seq.decoders.v2 import *

