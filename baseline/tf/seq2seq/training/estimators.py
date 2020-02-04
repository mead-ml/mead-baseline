import os
import tensorflow as tf
from eight_mile.tf.optz import optimizer
from baseline.tf.tfy import SET_TRAIN_FLAG
from baseline.train import register_training_func
from baseline.model import create_model_for
from baseline.tf.seq2seq.training.utils import to_tensors, SHUF_BUF_SZ, NUM_PREFETCH


def create_train_input_fn(ts, src_lengths_key, batchsz=1, gpus=1, **kwargs):
    """Creator function for an estimator to get a train dataset

    We use a closure to encapsulate the outer parameters

    :param ts: The data feed
    :param src_lengths_key: The key identifying the data feed field corresponding to length
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    tensors = to_tensors(ts, src_lengths_key)

    def train_input_fn():
        epochs = None
        train_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        train_dataset = train_dataset.batch(batchsz // gpus, drop_remainder=False)
        train_dataset = train_dataset.repeat(epochs)
        train_dataset = train_dataset.prefetch(NUM_PREFETCH)
        _ = train_dataset.make_one_shot_iterator()
        return train_dataset
    return train_input_fn


def create_valid_input_fn(vs, src_lengths_key, batchsz=1, **kwargs):
    """Creator function for an estimator to get a valid dataset

    We use a closure to encapsulate the outer parameters

    :param vs: The data feed
    :param src_lengths_key: The key identifying the data feed field corresponding to length
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param epochs: The number of epochs to train
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    tensors = to_tensors(vs, src_lengths_key)

    def eval_input_fn():
        valid_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
        valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)
        _ = valid_dataset.make_one_shot_iterator()
        return valid_dataset

    return eval_input_fn


def model_creator(model_params):
    """Create a model function which itself, if called yields a new model

    :param model_params: The parameters from MEAD for a model
    :return: A model function
    """
    def model_fn(features, labels, mode, params):
        """This function is used by estimators to create a model

        :param features: (`dict`) A dictionary of feature names mapped to tensors (iterators)
        :param labels: (`int`) These are the raw labels from file
        :param mode: (`str`): A TF mode including ModeKeys.PREDICT|TRAIN|EVAL
        :param params: A set of user-defined hyper-parameters passed by the estimator
        :return: A new model
        """
        model_params.update(features)
        model_params['sess'] = None

        # This is kind of crazy, but seems to work, hardwire a graph op for `mx_tgt_len`
        model_params.update({'tgt': labels, 'mx_tgt_len': tf.reduce_max(features['tgt_len'])})

        if mode == tf.estimator.ModeKeys.EVAL:
            SET_TRAIN_FLAG(False)
            model = create_model_for('seq2seq', **model_params)
            loss = model.create_loss()
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model.decoder.best, loss=loss)

        SET_TRAIN_FLAG(True)
        model = create_model_for('seq2seq', **model_params)
        loss = model.create_loss()
        colocate = True if params['gpus'] > 1 else False
        global_step, train_op = optimizer(loss,
                                          optim=params['optim'],
                                          eta=params.get('lr', params.get('eta')),
                                          colocate_gradients_with_ops=colocate)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.decoder.best,
                                          loss=loss,
                                          train_op=train_op)
    return model_fn


@register_training_func('seq2seq', 'estimator')
def fit_estimator(model_params, ts, vs, es=None, epochs=20, gpus=1, **kwargs):
    """Train the model with an `tf.estimator`

    To use this, you pass `fit_func: estimator` in your MEAD config

    This flavor of training utilizes both `tf.dataset`s and `tf.estimator`s to train.
    It is the preferred method for distributed training.

    FIXME: currently this method doesnt support early stopping.

    :param model_params: The model (or parameters to create the model) to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs:
        See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True
        * *verbose* (`dict`) A dictionary containing `console` boolean and `file` name if on
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *nsteps* (`int`) -- If we should report every n-steps, this should be passed
        * *ema_decay* (`float`) -- If we are doing an exponential moving average, what decay to us4e
        * *clip* (`int`) -- If we are doing gradient clipping, what value to use
        * *optim* (`str`) -- The name of the optimizer we are using
        * *lr* (`float`) -- The learning rate we are using
        * *mom* (`float`) -- If we are using SGD, what value to use for momentum
        * *beta1* (`float`) -- Adam-specific hyper-param, defaults to `0.9`
        * *beta2* (`float`) -- Adam-specific hyper-param, defaults to `0.999`
        * *epsilon* (`float`) -- Adam-specific hyper-param, defaults to `1e-8

    :return: None
    """
    model_fn = model_creator(model_params)
    #labels = model_params['labels']
    params = {
        #'labels': labels,
        'optim': kwargs['optim'],
        'lr': kwargs.get('lr', kwargs.get('eta')),
        'epochs': epochs,
        'gpus': gpus,
        'batchsz': kwargs['batchsz'],
        'test_batchsz': kwargs.get('test_batchsz', kwargs.get('batchsz'))
    }
    print(params)
    checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
    # We are only distributing the train function for now
    # https://stackoverflow.com/questions/52097928/does-tf-estimator-estimator-evaluate-always-run-on-one-gpu
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=gpus))
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)
    src_lengths_key = model_params.get('src_lengths_key')
    train_input_fn = create_train_input_fn(ts, src_lengths_key, **params)
    valid_input_fn = create_valid_input_fn(vs, src_lengths_key, **params)
    ##predict_input_fn = create_eval_input_fn(es, **params)

    # This is going to be None because train_and_evaluate controls the max steps so repeat doesnt matter
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=epochs * len(ts))
    # This is going to be None because the evaluation will run for 1 pass over the data that way
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
