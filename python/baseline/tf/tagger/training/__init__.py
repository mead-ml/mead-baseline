from eight_mile.utils import get_version
import tensorflow as tf
if get_version(tf) < 2:

    from baseline.tf.tagger.training.datasets import *
    from baseline.tf.tagger.training.feed import *
else:
    from baseline.tf.tagger.training.eager import *
