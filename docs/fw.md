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


In the TensorFlow backend, we use a global method function `TRAIN_FLAG()` to determine if things like dropout should be applied.  If the user is running `mead` to train (which is the typical case), this flag is automatically defined as a `tf.placeholder` that will default `False` (meaning no dropout will be applied).

This is convenient because it is possible to use the same graph definition for training and evaluation by simply re-defining the placeholder to `True` when we are training.

However, in some cases, we may wish to define multiple graphs, in which case this default behavior may not be suitable.  A typical example of this would be if the user is running a Baseline model outside of `mead` and wants to define separate graphs depending on what mode is running.  In this case, the `TRAIN_FLAG()` may be better suited as a boolean.  To facilitate this, we provide a method which allows the user to override the value of the `TRAIN_FLAG()` on demand.  This makes it possible to do code such as:

```
bl.tf.SET_TRAIN_FLAG(True)
model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, **model_params)
```

This is particularly helpful when using the `tf.estimator` API.  See [this example](../api-examples/tf-estimator.py) for details



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


#### Injecting inputs & avoiding placeholders in TensorFlow

The Baseline models are defined to support input inject, which bypasses placeholder creation.  This means that from custom user code, it is possible to directly inject the inputs. In the example below, we assume that embeddings is a dict containing the single key 'word', and we inject the input to that embedding via keyword arguments:

```
model = bl.model.create_model(embeddings, labels=params['labels'], word=word_source, y=y, **model_params)
```

This allows the user to access some of the advanced input capabilities in TensorFlow like `tf.dataset`s and `tf.Queue`s


## PyTorch
