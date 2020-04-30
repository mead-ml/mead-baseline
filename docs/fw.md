# Framework Specific Implementation Details

## TensorFlow

### Keras Layers vs TensorFlow

In MEAD with TensorFlow 1.x, most classes are implemented using Keras Layers underneath.
A major execption is RNNs -- for TF 1.x the default RNNs are implemented using the `tf.contrib.rnn` package, which we have found to be faster in 1.x than using the Keras equivalents, and which matches previous (MEAD 1.x) behavior.  In TensorFlow 1.x, its possible to use the Keras Layers using the suffix `2`, e.g. 

```
class LSTMEncoder2(tf.keras.layers.Layer)
```

In TensorFlow 2.x, we only use Keras Layers and it is not possible to use the `tf.contrib.rnn` since it is discontinued.

Under the hood, MEAD tests what version of TensorFlow is running and creates an alias from the proper class to e.g. `LSTMEncoder`.

### Dropout and the TRAIN_FLAG()

In early TensorFlow versions, it was quite common to construct the graph in code, whether or not it was to reload a checkpoint or to train a model.
Code that was executing for training typically would construct a different execution graph than training, to account for things like dropout.

MEAD/Baseline took a slightly different approach, defining a default placeholder for things like dropout which could default to off.
However, this made it hard to have lots of different types of dropouts on different sub-graphs, and we quickly realized that what we wanted
was more like what pytorch has -- a way of knowing if we are in training mode.  Whether we are in training is a global state, and so we just
have a variable (which is a default placeholder in declarative mode) called `TRAIN_FLAG()` that is applied to determine whether to turn on dropout.

In eager mode, there are no placeholders and we just want the `TRAIN_FLAG()` to be a boolean.  So we created a function `SET_TRAIN_FLAG()` which
can be set prior to invocation of `TRAIN_FLAG()` to set up the underlying boolean to a value `e.g False`.

The implementation is a little weird but it works:

```

def SET_TRAIN_FLAG(X):
    global BASELINE_TF_TRAIN_FLAG
    BASELINE_TF_TRAIN_FLAG = X


def TRAIN_FLAG():
    """Create a global training flag on first use"""
    global BASELINE_TF_TRAIN_FLAG
    if BASELINE_TF_TRAIN_FLAG is not None:
        return BASELINE_TF_TRAIN_FLAG

    BASELINE_TF_TRAIN_FLAG = tf.compat.v1.placeholder_with_default(False, shape=(), name="TRAIN_FLAG")
    return BASELINE_TF_TRAIN_FLAG

```


For TF version 1.x, the `smart_cond()` function makes this work no matter how `TRAIN_FLAG()` is defined:

```
 input_keep_prob = tf.contrib.framework.smart_cond(TRAIN_FLAG(), lambda: 1.0 - dropout, lambda: 1.0)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob)
```             

No matter if `TRAIN_FLAG()` is placeholder or a boolean, the above code always works!

Keras layers already handle this logic internally, so we dont use this in TensorFlow 2.x


training mode or inference.  This actually causes the graph to be identical as the placeholder itself changes the value of things like dropout.



## PyTorch
