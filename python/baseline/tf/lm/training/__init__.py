from eight_mile.utils import get_version
import tensorflow as tf
if get_version(tf) < 2:

    from baseline.tf.lm.training.datasets import *
    from baseline.tf.lm.training.estimators import *
    from baseline.tf.lm.training.feed import *
else:
    from baseline.tf.lm.training.eager import *
