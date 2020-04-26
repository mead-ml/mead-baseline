from eight_mile.utils import get_version
import tensorflow as tf
if not tf.executing_eagerly():

    from baseline.tf.lm.training.datasets import *
    from baseline.tf.lm.training.feed import *
else:
    from baseline.tf.lm.training.eager import *
    from baseline.tf.lm.training.distributed import *
