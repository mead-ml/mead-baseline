import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.utils import fill_y, listify, write_json, ls_props, read_json
from baseline.model import ClassifierModel, load_classifier_model, create_classifier_model
from baseline.tf.tfy import (stacked_lstm,
                             parallel_conv)

from baseline.version import __version__
import os
import copy


class ClassifyParallelModel(ClassifierModel):

    def __init__(self, create_fn, embeddings, labels, **kwargs):
        super(ClassifyParallelModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus', -1)
        # If the gpu ID is set to -1, use CUDA_VISIBLE_DEVICES to figure it out
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        print('Num GPUs', gpus)

        self.labels = labels
        nc = len(labels)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()
        for key in embeddings.keys():
            EmbeddingsType = embeddings[key].__class__
            self.parallel_params[key] = kwargs.get(key, EmbeddingsType.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus)

        self.lengths_key = kwargs.get('lengths_key')

        if self.lengths_key is not None:
            # This allows user to short-hand the field to use
            if not self.lengths_key.endswith('_lengths'):
                self.lengths_key += '_lengths'
            self.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths_parallel"))
            lengths_splits = tf.split(self.lengths, gpus)
            split_operations['lengths'] = lengths_splits

        else:
            self.lengths = None

        # This only exists to make exporting easier
        self.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y_parallel"))
        self.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        self.pdrop_value = kwargs.get('dropout', 0.5)

        y_splits = tf.split(self.y, gpus)
        split_operations['y'] = y_splits
        losses = []
        self.labels = labels

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) 
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            self.inference = create_fn(embeddings, labels, sess=sess, **kwargs)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                kwargs_single = copy.deepcopy(kwargs)
                kwargs_single['sess'] = sess
                kwargs_single['pkeep'] = self.pkeep

                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                replica = create_fn(embeddings, labels, **kwargs_single)
                self.replicas.append(replica)
                loss_op = replica.create_loss()
                losses.append(loss_op)

        self.loss = tf.reduce_mean(tf.stack(losses))

        self.sess = sess
        self.best = self.inference.best

    def create_loss(self):
        return self.loss

    def create_test_loss(self):
        return self.inference.create_test_loss()

    def save(self, model_base):
        return self.inference.save(model_base)

    def set_saver(self, saver):
        self.inference.saver = saver
        self.saver = saver

    def make_input(self, batch_dict, do_dropout=False):
        if do_dropout is False:
            return self.inference.make_input(batch_dict)

        y = batch_dict.get('y', None)
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.parallel_params.keys():
            feed_dict["{}_parallel:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = fill_y(len(self.labels), y)
        return feed_dict


class ClassifierModelBase(ClassifierModel):
    """Base for all baseline implementations of token-based classifiers
    
    This class provides a loose skeleton around which the baseline models
    are built.  This essentially consists of dividing up the network into a logical separation between "embedding",
    or composition of lookup tables to build a vector representation of a temporal input, "pooling",
    or the conversion of temporal data to a fixed representation, and "stacking" layers, which are (optional)
    fully-connected layers below, followed by a projection to output space and a softmax
    
    For instance, the baseline convolutional and LSTM models implement pooling as CMOT, and LSTM last time
    respectively, whereas, neural bag-of-words (NBoW) do simple max or mean pooling followed by multiple fully-
    connected layers.
    
    """
    def __init__(self):
        """Base
        """
        super(ClassifierModelBase, self).__init__()

    def set_saver(self, saver):
        self.saver = saver

    def save_values(self, basename):
        """Save tensor files out

        :param basename: Base name of model
        :return:
        """
        self.saver.save(self.sess, basename)

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """
        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        # For each embedding, save a record of the keys

        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__
        state = {
            "version": __version__,
            "embeddings": embeddings_info,
            "lengths_key": self.lengths_key
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        write_json(self.labels, basename + ".labels")
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

    def save(self, basename, **kwargs):
        """Save meta-data and actual data for a model

        :param basename: (``str``) The model basename
        :param kwargs:
        :return: None
        """
        self.save_md(basename)
        self.save_values(basename)

    def create_test_loss(self):
        with tf.name_scope("test_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def create_loss(self):
        """The loss function is currently provided here, although this is not a great place for it
        as it provides a coupling between the model and its loss function.  Just here for convenience at the moment.
        
        :return: 
        """
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def classify(self, batch_dict):
        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output. You can use this method directly if you have vector input, or you can use the `ClassifierService`
        which can convert directly from text durign its `transform`.  That method calls this one underneath.
        
        :param batch_dict: (``dict``) Contains any inputs to embeddings for this model
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """
        feed_dict = self.make_input(batch_dict)
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, do_dropout=False):
        """Transform a `batch_dict` into a TensorFlow `feed_dict`

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :param do_dropout: (``bool``) Should we do dropout.  Defaults to False
        :return:
        """
        y = batch_dict.get('y', None)

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = fill_y(len(self.labels), y)
        return feed_dict

    def get_labels(self):
        """Get the string labels back
        
        :return: labels
        """
        return self.labels

    @classmethod
    def load(cls, basename, **kwargs):
        """Reload the model from a graph file and a checkpoint
        
        The model that is loaded is independent of the pooling and stacking layers, making this class reusable
        by sub-classes.
        
        :param basename: The base directory to load from
        :param kwargs: See below
        
        :Keyword Arguments:
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created
        
        :return: A restored model
        """
        sess = kwargs.get('session', kwargs.get('sess', tf.Session()))
        model = cls()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename

        state = read_json(basename + '.state')

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            try:
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            except:
                # Backwards compat
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name + ".model"})

        model.embeddings = dict()
        for key, class_name in state['embeddings'].items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            embed_args[key] = tf.get_default_graph().get_tensor_by_name('{}:0'.format(key))
            Constructor = eval(class_name)
            model.embeddings[key] = Constructor(key, **embed_args)

        model.lengths_key = state.get('lengths_key')
        if model.lengths_key is not None:
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
        else:
            model.lengths = None
        model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
        model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
        model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')

        model.labels = read_json(basename + '.labels')
        model.sess = sess
        return model

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        """The main method for creating all :class:`WordBasedModel` types.
        
        This method instantiates a model with pooling and optional stacking layers.
        Many of the arguments provided are reused by each implementation, but some sub-classes need more
        information in order to properly initialize.  For this reason, the full list of keyword args are passed
        to the :method:`pool` and :method:`stacked` methods.
        
        :param embeddings: This is a dictionary of embeddings, mapped to their numerical indices in the lookup table
        :param labels: This is a list of the `str` labels
        :param kwargs: There are sub-graph specific Keyword Args allowed for e.g. embeddings. See below for known args:
        
        :Keyword Arguments:
        * *gpus* -- (``int``) How many GPUs to split training across.  If called this function delegates to
            another class `ClassifyParallelModel` which creates a parent graph and splits its inputs across each
            sub-model, by calling back into this exact method (w/o this argument), once per GPU
        * *model_type* -- The string name for the model (defaults to `default`)
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created
        * *lengths_key* -- (``str``) Specifies which `batch_dict` property should be used to determine the temporal length
            if this is not set, it defaults to either `word`, or `x` if `word` is also not a feature
        * *finetune* -- Are we doing fine-tuning of word embeddings (defaults to `True`)
        * *mxlen* -- The maximum signal (`x` tensor temporal) length (defaults to `100`)
        * *dropout* -- This indicates how much dropout should be applied to the model when training.
        * *pkeep* -- By default, this is a `tf.placeholder`, but it can be passed in as part of a sub-graph.
            This is useful for exporting tensorflow models or potentially for using input tf queues
        * *filtsz* -- This is actually a top-level param due to an unfortunate coupling between the pooling layer
            and the input, which, for convolution, requires input padding.
        
        :return: A fully-initialized tensorflow classifier 
        """

        gpus = kwargs.get('gpus')
        # If we are parallelized, we will use the wrapper object ClassifyParallelModel and this creation function
        if gpus is not None:
            return ClassifyParallelModel(cls.create, embeddings, labels, **kwargs)
        sess = kwargs.get('sess', tf.Session())

        model = cls()
        model.embeddings = embeddings
        model.lengths_key = kwargs.get('lengths_key')
        if model.lengths_key is not None:
            # This allows user to short-hand the field to use
            if not model.lengths_key.endswith('_lengths'):
                model.lengths_key += '_lengths'
            model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        else:
            model.lengths = None

        model.labels = labels
        nc = len(labels)
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            seed = np.random.randint(10e8)
            init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
            xavier_init = xavier_initializer(True, seed)
            word_embeddings = model.embed()
            input_sz = word_embeddings.shape[-1]
            pooled = model.pool(word_embeddings, input_sz, init, **kwargs)
            stacked = model.stacked(pooled, init, **kwargs)

            # For fully connected layers, use xavier (glorot) transform
            with tf.contrib.slim.arg_scope(
                    [fully_connected],
                    weights_initializer=xavier_init):
                with tf.variable_scope("output"):
                    model.logits = tf.identity(fully_connected(stacked, nc, activation_fn=None), name="logits")
                    model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        # writer = tf.summary.FileWriter('blah', sess.graph)
        return model

    def embed(self):
        """Thie method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        for embedding in self.embeddings.values():
            embeddings_out = embedding.encode()
            all_embeddings_out += [embeddings_out]
        word_embeddings = tf.concat(values=all_embeddings_out, axis=2)
        return word_embeddings

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """This method performs a transformation between a temporal signal and a fixed representation
        
        :param word_embeddings: The output of the embedded lookup, which is the starting point for this operation
        :param dsz: The depth of the embeddings
        :param init: The tensorflow initializer to use for these methods
        :param kwargs: Model-specific arguments
        :return: A fixed representation of the data
        """
        pass

    def stacked(self, pooled, init, **kwargs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param pooled: The fixed representation of the model
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """

        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            return pooled

        in_layer = pooled
        for i, hsz in enumerate(hszs):
            with tf.variable_scope('fc-{}'.format(i)):
                with tf.contrib.slim.arg_scope(
                        [fully_connected],
                        weights_initializer=init):
                    fc = fully_connected(in_layer, hsz, activation_fn=tf.nn.relu)
                    in_layer = tf.nn.dropout(fc, self.pkeep)
        return in_layer


class ConvModel(ClassifierModelBase):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width
    
    """
    def __init__(self):
        """Constructor 
        """
        super(ConvModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling
        
        :param word_embeddings: The word embeddings, which are inputs here
        :param dsz: The depth of the word embeddings
        :param init: The tensorflow initializer
        :param kwargs: See below
        
        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use
        
        
        :return: 
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']

        combine, _ = parallel_conv(word_embeddings, filtsz, dsz, cmotsz)
        # Definitely drop out
        with tf.name_scope("dropout"):
            combine = tf.nn.dropout(combine, self.pkeep)
        return combine


class LSTMModel(ClassifierModelBase):
    """A simple single-directional single-layer LSTM. No layer-stacking.
    
    """
    def __init__(self):
        super(LSTMModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """LSTM with dropout yielding a final-state as output
        
        :param word_embeddings: The input word embeddings
        :param dsz: The input word embedding depth
        :param init: The tensorflow initializer to use (currently ignored)
        :param kwargs: See below
        
        :Keyword Arguments:
        * *rnnsz* -- (``int``) The number of hidden units (defaults to `hsz`)
        * *hsz* -- (``int``) backoff for `rnnsz`, typically a result of stacking params.  This keeps things simple so
          its easy to do things like residual connections between LSTM and post-LSTM stacking layers
        
        :return: 
        """
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]

        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        nlayers = int(kwargs.get('layers', 1))

        if rnntype == 'blstm':
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            rnnbwd = stacked_lstm(hsz, self.pkeep, nlayers)
            ((_, _), (fw_final_state, bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(rnnfwd,
                                                                                         rnnbwd,
                                                                                         word_embeddings,
                                                                                         sequence_length=self.lengths,
                                                                                         dtype=tf.float32)
            # The output of the BRNN function needs to be joined on the H axis
            output_state = fw_final_state[-1].h + bw_final_state[-1].h
            out_hsz = hsz

        else:
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            (_, (output_state)) = tf.nn.dynamic_rnn(rnnfwd, word_embeddings, sequence_length=self.lengths, dtype=tf.float32)
            output_state = output_state[-1].h
            out_hsz = hsz

        combine = tf.reshape(output_state, [-1, out_hsz])
        return combine


class NBowBase(ClassifierModelBase):
    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """
    def __init__(self):
        super(NBowBase, self).__init__()

    def stacked(self, pooled, init, **kwargs):
        """Force at least one hidden layer here

        :param pooled:
        :param init:
        :param kwargs:
        :return:
        """
        kwargs['hsz'] = kwargs.get('hsz', [100])
        return super(NBowBase, self).stacked(pooled, init, **kwargs)


class NBowModel(NBowBase):
    """Neural Bag-of-Words average pooling (standard) model"""
    def __init__(self):
        super(NBowModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do average pooling on input embeddings, yielding a `dsz` output layer
        
        :param word_embeddings: The word embedding input
        :param dsz: The word embedding depth
        :param init: The tensorflow initializer
        :param kwargs: None
        :return: The average pooling representation
        """
        return tf.reduce_mean(word_embeddings, 1, keepdims=False)


class NBowMaxModel(NBowBase):
    """Max-pooling model for Neural Bag-of-Words.  Sometimes does better than avg pooling
    """
    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do max pooling on input embeddings, yielding a `dsz` output layer
        
        :param word_embeddings: The word embedding input
        :param dsz: The word embedding depth
        :param init: The tensorflow initializer
        :param kwargs: None
        :return: The max pooling representation
        """
        return tf.reduce_max(word_embeddings, 1, keepdims=False)


class CompositePoolingModel(ClassifierModelBase):
    """Fulfills pooling contract by aggregating pooling from a set of sub-models and concatenates each
    """
    def __init__(self):
        """
        Construct a composite pooling model
        """
        super(CompositePoolingModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Cycle each sub-model and call its pool method, then concatenate along final dimension

        :param word_embeddings: The input graph
        :param dsz: The number of input units
        :param init: The initializer operation
        :param kwargs:
        :return: A pooled composite output
        """
        SubModels = [eval(model) for model in kwargs.get('sub')]
        pooling = []
        for SubClass in SubModels:
            pooling += [SubClass.pool(self, word_embeddings, dsz, init, **kwargs)]
        return tf.concat(pooling, -1)


BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create,
    'composite': CompositePoolingModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMaxModel.load
}


def create_model(embeddings, labels, **kwargs):
    """This function creates a classifier with known embeddings and labels using the `model_type`.
    If the model is found in a list of known models (keyed off `model_type`), it is constructed using a `create`
    method known a priori (e.g. `LSTMModel.create`).  If the `mode_type` is a name not found in the dict of known
    models, try to find a `classify_{model_type}` in the `PYTHONPATH` and load that instead.  This is a plugin facility
    that allows baseline to be extended with custom models (a common use-case)

    :param embeddings: (``dict``) A dictionary of embeddings sub-graphs or models
    :param labels: A set of labels
    :param kwargs: Addon models may have arbitary keyword args.  The known arguments are listed below

    :Keyword Arguments:
    * *model_type* - (``str``) The key name of this model.  If its not found, we go looking for an addon in the
      PYTHONPATH and load that module
    :return: A model
    """
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    """This function loads a classifier with known embeddings and labels using the `model_type`.
    If the model is found in a list of known models (keyed off `model_type`), it is constructed using a `load`
    method known a priori (e.g. `LSTMModel.create`).  If the `mode_type` is a name not found in the dict of known
    models, try to find a `classify_{model_type}` in the `PYTHONPATH` and load that instead.  This is a plugin facility
    that allows baseline to be extended with custom models (a common use-case)

    :param embeddings: (``dict``) A dictionary of embeddings sub-graphs or models
    :param labels: A set of labels
    :param kwargs: Addon models may have arbitary keyword args.  The known arguments are listed below

    :Keyword Arguments:
    * *model_type* - (``str``) The key name of this model.  If its not found, we go looking for an addon in the
      PYTHONPATH and load that module
    :return: A model
    """
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)
