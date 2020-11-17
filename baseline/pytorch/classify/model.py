import logging
from baseline.model import ClassifierModel, register_model
from baseline.pytorch.torchy import *
from baseline.utils import listify, write_json
from eight_mile.pytorch.layers import *
import torch.backends.cudnn as cudnn
import os
cudnn.benchmark = True

logger = logging.getLogger('baseline')


class ClassifierModelBase(nn.Module, ClassifierModel):
    """Base for all baseline implementations of token-based classifiers

    This class provides a loose skeleton around which the baseline models
    are built.  It is built on the PyTorch `nn.Module` base, and fulfills the `ClassifierModel` interface.
    To override this class, the use would typically override the `create_layers` function which will
    create and attach all sub-layers of this model into the class, and the `forward` function which will
    give the model some implementation to call on forward.
    """

    def __init__(self):
        super().__init__()
        self.gpu = False

    @classmethod
    def load(cls, filename: str, **kwargs) -> 'ClassifierModelBase':
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def save(self, outname: str):
        logger.info('saving %s' % outname)
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)
        write_json(self.labels, basename + ".labels")

    @classmethod
    def create(cls, embeddings, labels, **kwargs) -> 'ClassifierModelBase':

        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.5)
        model.lengths_key = kwargs.get('lengths_key')
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels
        model.create_layers(embeddings, **kwargs)
        logger.info(model)
        return model

    def cuda(self, device=None):
        self.gpu = True
        return super().cuda(device=device)

    def create_loss(self):
        return nn.NLLLoss()

    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):

        """Transform a `batch_dict` into something usable in this model

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :return:
        """
        example_dict = dict({})
        perm_idx = None

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            lengths = batch_dict[self.lengths_key]
            if numpy_to_tensor:
                lengths = torch.from_numpy(lengths)
            lengths, perm_idx = lengths.sort(0, descending=True)
            if self.gpu:
                lengths = lengths.cuda()
            example_dict['lengths'] = lengths

        for key in self.embeddings.keys():
            tensor = batch_dict[key]
            if numpy_to_tensor:
                tensor = torch.from_numpy(tensor)
            if perm_idx is not None:
                tensor = tensor[perm_idx]
            if self.gpu:
                tensor = tensor.cuda()
            example_dict[key] = tensor

        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if perm_idx is not None:
                y = y[perm_idx]
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        if perm:
            return example_dict, perm_idx

        return example_dict

    def predict_batch(self, batch_dict: Dict[str, TensorDef], **kwargs) -> TensorDef:
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        examples, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        with torch.no_grad():
            probs = self(examples).exp()
            probs = unsort_batch(probs, perm_idx)
        return probs

    def predict(self, batch_dict: Dict[str, TensorDef], raw: bool = False, dense: bool = False, **kwargs):
        self.eval()
        probs = self.predict_batch(batch_dict, **kwargs)
        if raw and not dense:
            logger.warning(
                "Warning: `raw` parameter is deprecated pass `dense=True` to get back values as a single tensor")

            dense = True
        if dense:
            return probs
        results = []
        batchsz = probs.size(0)
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def get_labels(self) -> List[str]:
        return self.labels

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """


class EmbedPoolStackClassifier(ClassifierModelBase):

    """Provides a simple but effective base for most `ClassifierModel`s

   This class provides a common base for classifiers by identifying and codifying
   and idiomatic pattern where a typical classifier may be though of as a composition
   between a stack of embeddings, followed by a pooling operation yielding a fixed length
   tensor, followed by one or more dense layers, and ultimately, a projection to the output space.

   To provide an useful interface to sub-classes, we override the `create_layers` to provide a hook
   for each layer identified in this idiom, and leave the implementations up to the sub-class.

   We also fully implement the `forward` method.

   """

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        self.stack_model = self.init_stacked(self.pool_model.output_dim, **kwargs)
        self.output_layer = self.init_output(self.stack_model.output_dim, **kwargs)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a pooling operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A pooling operation
        """

    def init_stacked(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop)

    def init_output(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce the final output layer in the model

        :param input_dim: The input hidden size
        :param kwargs:
        :return:
        """
        return Dense(input_dim, len(self.labels), activation=kwargs.get('output_activation', 'log_softmax'))

    def forward(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """
        lengths = inputs.get("lengths")
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


@register_model(task='classify', name='default')
class ConvModel(EmbedPoolStackClassifier):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width
    """

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling

        :param input_dim: Embedding output size
        :param kwargs: See below

        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use

        :return: A pooling layer
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        return WithoutLength(WithDropout(ParallelConv(input_dim, cmotsz, filtsz, "relu", input_fmt="bth"), self.pdrop))


@register_model(task='classify', name='lstm')
class LSTMModel(EmbedPoolStackClassifier):
    """A simple single-directional single-layer LSTM. No layer-stacking.
    """

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """LSTM with dropout yielding a final-state as output

        :param input_dim: The input word embedding depth
        :param kwargs: See below

        :Keyword Arguments:
        * *rnnsz* -- (``int``) The number of hidden units (defaults to `hsz`)
        * *rnntype/rnn_type* -- (``str``) The RNN type, defaults to `lstm`, other valid values: `blstm`
        * *hsz* -- (``int``) backoff for `rnnsz`, typically a result of stacking params.  This keeps things simple so
          its easy to do things like residual connections between LSTM and post-LSTM stacking layers

        :return: A pooling layer
        """
        unif = kwargs.get('unif')
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        weight_init = kwargs.get('weight_init', 'uniform')
        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        if rnntype == 'blstm':
            return BiLSTMEncoderHidden(input_dim, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)
        return LSTMEncoderHidden(input_dim, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)


class NBowModelBase(EmbedPoolStackClassifier):
    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """

    def init_stacked(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model, defaulting to a single layer

        :param input_dim: The input dimension size
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``List[int]``) The number of hidden units (defaults to 100)
        """
        kwargs.setdefault('hsz', [100])
        return super().init_stacked(input_dim, **kwargs)


@register_model(task='classify', name='nbow')
class NBowModel(NBowModelBase):
    """Neural Bag-of-Words average pooling (standard) model"""

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Do average pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: None
        :return: The average pooling representation
        """
        return MeanPool1D(input_dim)


@register_model(task='classify', name='nbowmax')
class NBowMaxModel(NBowModelBase):
    """Max-pooling model for Neural Bag-of-Words.  Sometimes does better than avg pooling
    """

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Do max pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: None
        :return: The max pooling representation
        """
        return MaxPool1D(input_dim)


@register_model(task='classify', name='fine-tune')
class FineTuneModelClassifier(ClassifierModelBase):
    """Fine-tune based on pre-pooled representations"""

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def init_stacked(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop)

    def init_output(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce the final output layer in the model

        :param input_dim: The input hidden size
        :param kwargs:
        :return:
        """
        return WithDropout(Dense(input_dim, len(self.labels), activation=kwargs.get('output_activation', 'log_softmax'),
                                 unif=kwargs.get('output_unif', 0.0)), pdrop=kwargs.get('output_dropout', 0.0))

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.stack_model = self.init_stacked(self.embeddings.output_dim, **kwargs)
        self.output_layer = self.init_output(self.stack_model.output_dim, **kwargs)

    def forward(self, inputs):
        base_layers = self.embeddings(inputs)
        stacked = self.stack_model(base_layers)
        return self.output_layer(stacked)


@register_model(task='classify', name='fine-tune-paired')
class FineTunePairedClassifierModel(FineTuneModelClassifier):

    """Fine-tuning model for pairs

    This model encodes a pair as a single utterance using some encoding scheme defined in
    ``_convert_pair`` which is fed directly into the fine-tuning model.

    For BERT, this simply encodes the input key pair as a single utterance while building
    a token-type vector.

    For the input, we will assume that the vectorizer will be producing a start token and an end token.
    We will simply remove the start token from the second sentence and concatenate
    [CLS] this is sentence one [SEP]

    [CLS] this is sentence two [SEP]


    """
    def _convert_pair(self, key, batch_dict, numpy_to_tensor, example_dict):

        toks = batch_dict[f"{key}"]
        token_type_key = f"{key}_tt"
        tt = batch_dict.get(token_type_key)
        if numpy_to_tensor:
            toks = torch.from_numpy(toks)
            if tt is not None:
                tt = torch.from_numpy(tt)
        if self.gpu:
            toks = toks.cuda()
            if tt is not None:
                tt = tt.cuda()
        if tt is not None:
            example_dict[key] = (toks, tt)
        else:
            example_dict[key] = toks

    def make_input(self, batch_dict: Dict, perm: bool = False, numpy_to_tensor: bool = False):

        """Transform a `batch_dict` into something usable in this model

        :param batch_dict: A dictionary containing all inputs to the embeddings for this model
        :param perm: Should we permute the batch, this is not supported here
        :param numpy_to_tensor: Do we need to convert from numpy to a ``torch.Tensor``
        :return: An example dictionary for processing
        """
        example_dict = dict({})
        # Allow us to track a length, which is needed for BLSTMs
        for key in self.embeddings.keys():
            self._convert_pair(key, batch_dict, numpy_to_tensor, example_dict)

        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        return example_dict


@register_model(task='classify', name='composite')
class CompositePoolingModel(ClassifierModelBase):
    """Fulfills pooling contract by aggregating pooling from a set of sub-models and concatenates each"""

    def init_pool(self, dsz, **kwargs):
        SubModels = [eval(model) for model in kwargs.get('sub')]
        sub_models = [SM.init_pool(self, dsz, **kwargs) for SM in SubModels]
        return CompositePooling(sub_models)

    def make_input(self, batch_dict):
        """Because the sub-model could contain an LSTM, make sure to sort lengths descending

        :param batch_dict:
        :return:
        """
        inputs = super().make_input(batch_dict)
        lengths = inputs['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        for k, value in inputs.items():
            inputs[k] = value[perm_idx]
        return inputs

