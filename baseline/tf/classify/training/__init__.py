from eight_mile.utils import get_version
import tensorflow as tf
if not tf.executing_eagerly():

    from baseline.tf.classify.training.datasets import *
    from baseline.tf.classify.training.feed import *
else:
    from baseline.tf.classify.training.eager import *
    from baseline.tf.classify.training.distributed import *
