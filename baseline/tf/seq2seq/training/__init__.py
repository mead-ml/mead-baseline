from eight_mile.utils import get_version
import tensorflow as tf
if get_version(tf) < 2:

    from baseline.tf.seq2seq.training.datasets import *
    from baseline.tf.seq2seq.training.estimators import *
    from baseline.tf.seq2seq.training.feed import *
else:
    from baseline.tf.seq2seq.training.eager import *
