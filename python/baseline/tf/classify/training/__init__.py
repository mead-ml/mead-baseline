from eight_mile.utils import get_version
import tensorflow as tf
if get_version(tf) < 2:

    from baseline.tf.classify.training.datasets import *
    from baseline.tf.classify.training.estimators import *
    from baseline.tf.classify.training.feed import *
else:
    from baseline.tf.classify.training.eager import *