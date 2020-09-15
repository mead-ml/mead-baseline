from eight_mile.utils import get_version
import tensorflow as tf
if not tf.executing_eagerly():
    raise Exception("Non-eager dependency parsing is currently unsupported")
else:
    from baseline.tf.deps.training.eager import *
