from eight_mile.utils import get_version
import tensorflow as tf
if not tf.executing_eagerly():
    from baseline.tf.tagger.training.feed import *
else:
    from baseline.tf.tagger.training.eager import *
