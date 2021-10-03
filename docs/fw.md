# Framework Specific Implementation Details

## TensorFlow

### Dropout and the TRAIN_FLAG()


In the TensorFlow backend, we use a global method function `TRAIN_FLAG()` to determine if things like dropout should be applied.  If the user is running `mead` to train (which is the typical case), this flag is automatically defined as a boolean that will default `False` (meaning no dropout will be applied).

We provide a method which allows the user to override the value of the `TRAIN_FLAG()` on demand.

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



