import logging
from baseline.pytorch.torchy import *
from baseline.utils import Offsets, write_json
from baseline.model import TaggerModel
from baseline.model import register_model
import torch.autograd
import os

logger = logging.getLogger('baseline')


class TaggerModelBase(nn.Module, TaggerModel):
    """Base class for tagger models

    This class provides the model base for tagging.  To create a tagger, overload `create_layers()` and `forward()`.
    Most implementations should be able to subclass the `AbstractEncoderTaggerModel`, which inherits from this and imposes
    additional structure
    """
    def __init__(self):
        """Constructor"""
        super().__init__()
        self.gpu = False

    def save(self, outname: str):
        """Save out the model

        :param outname: The name of the checkpoint to write
        :return:
        """
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)
        write_json(self.labels, basename + ".labels")

    def cuda(self, device=None):
        self.gpu = True
        return super().cuda(device=device)


    @staticmethod
    def load(filename: str, **kwargs) -> 'TaggerModelBase':
        """Create and load a tagger model from file

        """
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def drop_inputs(self, key, x):
        """Do dropout on inputs, using the dropout value (or none if not set)
        This works by applying a dropout mask with the probability given by a
        value within the `dropin_values: Dict[str, float]`, keyed off the text name
        of the feature

        :param key: The feature name
        :param x: The tensor to drop inputs for
        :return: The dropped out tensor
        """
        v = self.dropin_values.get(key, 0)
        if not self.training or v == 0:
            return x

        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).to(mask_pad.dtype)
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx, numpy_to_tensor=False):
        """Given a batch of input, and a key, prepare and noise the input

        :param key: The key of the tensor
        :param batch_dict: The batch of data as a dictionary of tensors
        :param perm_idx: The proper permutation order to get lengths descending
        :param numpy_to_tensor: Should this tensor be converted to a `torch.Tensor`
        :return:
        """
        tensor = batch_dict[key]
        if numpy_to_tensor:
            tensor = torch.from_numpy(tensor)

        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def make_input(self, batch_dict: Dict[str, TensorDef], perm: bool = False, numpy_to_tensor: bool = False) -> Dict[str, TensorDef]:
        """Transform a `batch_dict` into format suitable for tagging

        :param batch_dict: A dictionary containing all inputs to the embeddings for this model
        :param perm: Should we sort data by length descending?
        :param numpy_to_tensor: Do we need to convert the input from numpy to a torch.Tensor?
        :return: A dictionary representation of this batch suitable for processing
        """
        example_dict = dict({})
        lengths = batch_dict[self.lengths_key]
        if numpy_to_tensor:
            lengths = torch.from_numpy(lengths)

        lengths, perm_idx = lengths.sort(0, descending=True)
        if self.gpu:
            lengths = lengths.cuda()

        example_dict['lengths'] = lengths
        for key in self.embeddings.keys():
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx, numpy_to_tensor=numpy_to_tensor)
        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            y = y[perm_idx]
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        ids = batch_dict.get('ids')
        if ids is not None:
            if numpy_to_tensor:
                ids = torch.from_numpy(ids)
            ids = ids[perm_idx]
            if self.gpu:
                ids = ids.cuda()
            example_dict['ids'] = ids
        if perm:
            return example_dict, perm_idx
        return example_dict

    def get_labels(self) -> List[str]:
        """Get the labels (names of each class)

        :return: (`List[str]`) The labels
        """
        return self.labels

    def predict(self, batch_dict: Dict[str, TensorDef], **kwargs) -> TensorDef:
        """Take in a batch of data, and predict the tags

        :param batch_dict: A batch of features that is to be predicted
        :param kwargs: See Below

        :Keyword Arguments:

        * *numpy_to_tensor* (``bool``) Should we convert input from numpy to `torch.Tensor` Defaults to `True`
        :return: A batch-sized tensor of predictions
        """
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        inputs, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        outputs = self(inputs)
        return unsort_batch(outputs, perm_idx)

    @classmethod
    def create(cls, embeddings: Dict[str, TensorDef], labels: List[str], **kwargs) -> 'TaggerModelBase':
        """Create a tagger from the inputs.  Most classes shouldnt extend this

        :param embeddings: A dictionary containing the input feature indices
        :param labels: A list of the labels (tags)
        :param kwargs: See below

        :Keyword Arguments:

        * *lengths_key* (`str`) Which feature identifies the length of the sequence
        * *activation* (`str`) What type of activation function to use (defaults to `tanh`)
        * *dropout* (`str`) What fraction dropout to apply
        * *dropin* (`str`) A dictionarwith feature keys telling what fraction of word masking to apply to each feature

        :return:
        """
        model = cls()
        model.lengths_key = kwargs.get('lengths_key')
        model.activation_type = kwargs.get('activation', 'tanh')
        model.pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.create_layers(embeddings, **kwargs)
        return model

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """
    def compute_loss(self, inputs):
        """Define a loss function from the inputs, which includes the gold tag values as `inputs['y']`

        :param inputs:
        :return:
        """


class AbstractEncoderTaggerModel(TaggerModelBase):
    """Class defining a typical flow for taggers.  Most taggers should extend this class

    This class provides the model base for tagging by providing specific hooks for each phase.  There are
    4 basic steps identified in this class:

    1. embed
    2. encode (transduction)
    3. proj (projection to the final number of labels)
    4. decode

    There is an `init_* method for each of this phases, allowing you to
    define and return a custom layer.

    The actual forward method is defined as a combination of these 3 steps, which includes a
    projection from the encoder output to the number of labels.

    Decoding in taggers refers to the process of selecting the best path through the labels and is typically
    implemented either as a constrained greedy decoder or as a CRF layer
    """
    def __init__(self):
        """Constructor"""
        super().__init__()

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        return EmbeddingsStack(embeddings, self.pdrop, reduction=kwargs.get('embeddings_reduction', 'concat'))

    def init_encode(self, input_dim, **kwargs) -> BaseLayer:
        """Provide a layer object that represents the `encode` phase of the model
        :param input_dim: The hidden input size
        :param kwargs:
        :return: The encoder
        """

    def init_proj(self, **kwargs) -> BaseLayer:
        """Provide a projection from the encoder output to the number of labels

        This projection typically will not include any activation, since its output is the logits that
        the decoder is built on

        :param kwargs:
        :return: A projection from the encoder output size to the final number of labels
        """
        return Dense(self.encoder.output_dim, len(self.labels))

    def init_decode(self, **kwargs) -> BaseLayer:
        """Define a decoder from the inputs

        :param kwargs: See below
        :keyword Arguments:
        * *crf* (``bool``) Should we create a CRF as the decoder
        * *constraint_mask* (``tensor``) A constraint mask to apply to the transitions
        * *reduction* (``str``) How to reduce the loss, defaults to `batch`
        :return: A decoder layer
        """
        use_crf = bool(kwargs.get('crf', False))
        constraint_mask = kwargs.get('constraint_mask')
        if constraint_mask is not None:
            constraint_mask = constraint_mask.unsqueeze(0)

        if use_crf:
            decoder = CRF(len(self.labels), constraint_mask=constraint_mask, batch_first=True)
        else:
            decoder = TaggerGreedyDecoder(
                len(self.labels),
                constraint_mask=constraint_mask,
                batch_first=True,
                reduction=kwargs.get('reduction', 'batch')
            )
        return decoder

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This class overrides this method to produce the outline of steps for a transduction tagger

        :param embeddings: The input embeddings dict
        :param kwargs:
        :return:
        """
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.encoder = self.init_encode(self.embeddings.output_dim, **kwargs)
        self.proj_layer = self.init_proj(**kwargs)
        self.decoder = self.init_decode(**kwargs)

    def transduce(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """This operation performs embedding of the input, followed by encoding and projection to logits

        :param inputs: The feature indices to embed
        :return: Transduced (post-encoding) output
        """
        lengths = inputs["lengths"]
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        transduced = self.proj_layer(self.encoder(embedded))
        return transduced

    def decode(self, tensor: TensorDef, lengths: TensorDef) -> TensorDef:
        """Take in the transduced (encoded) input and decode it

        :param tensor: Transduced input
        :param lengths: Valid lengths of the transduced input
        :return: A best path through the output
        """
        return self.decoder((tensor, lengths))

    def forward(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Take the input and produce the best path of labels out

        :param inputs: The feature indices for the input
        :return: The most likely path through the output labels
        """
        transduced = self.transduce(inputs)
        path = self.decode(transduced, inputs.get("lengths"))
        return path

    def compute_loss(self, inputs):
        """Provide the loss by requesting it from the decoder

        :param inputs: A batch of inputs
        :return:
        """
        tags = inputs['y']
        lengths = inputs['lengths']
        unaries = self.transduce(inputs)
        return self.decoder.neg_log_loss(unaries, tags, lengths)


@register_model(task='tagger', name='default')
class RNNTaggerModel(AbstractEncoderTaggerModel):
    """RNN-based tagger implementation: this is the default tagger for mead-baseline

    Overload the encoder, typically as a BiLSTM
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) -> BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *rnntype* (``str``) The type of RNN, defaults to `blstm`
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate
        * *weight_init* (``str``) The weight initializer, defaults to `uniform`
        * *unif* (``float``) A value for the weight initializer
        :return: An encoder
        """
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = int(kwargs.get('layers', 1))
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        pdrop = float(kwargs.get('dropout', 0.5))
        weight_init = kwargs.get('weight_init', 'uniform')
        Encoder = LSTMEncoderSequence if rnntype == 'lstm' else BiLSTMEncoderSequence
        return Encoder(input_dim, hsz, nlayers, pdrop, unif=unif, initializer=weight_init, batch_first=True)


@register_model(task='tagger', name='transformer')
class TransformerTaggerModel(AbstractEncoderTaggerModel):
    """Transformer-based tagger model

    Overload the encoder using a length-aware Transformer
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) -> BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *num_heads* (``int``) The number of heads for multi-headed attention
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *d_ff* (``int``) The feed-forward layer size
        * *rpr_k* (``list`` or ``int``) The relative attention sizes.  If its a list, one scalar per layer, if its
          a scalar, apply same size to each layer
        :return: An encoder
        """
        layers = int(kwargs.get('layers', 1))
        num_heads = int(kwargs.get('num_heads', 4))
        pdrop = float(kwargs.get('dropout', 0.5))
        scale = False
        hsz = int(kwargs['hsz'])
        rpr_k = kwargs.get('rpr_k', 100)
        d_ff = kwargs.get('d_ff')
        encoder = TransformerEncoderStackWithLengths(num_heads, hsz, pdrop, scale, layers, d_ff=d_ff, rpr_k=rpr_k, input_sz=input_dim)
        return encoder


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(AbstractEncoderTaggerModel):
    """Convolutional (AKA TDNN) tagger

    Overload the encoder using a conv layer

    """
    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) -> BaseLayer:
        """Override the base method to produce an RNN transducer

        :param input_dim: The size of the input
        :param kwargs: See below

        :Keyword Arguments:
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *activation_type* (``str``) Defaults to `relu`
        * *wfiltsz* (``int``) The 1D filter size for the convolution
        :return: An encoder
        """
        layers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        filtsz = kwargs.get('wfiltsz', 5)
        activation_type = kwargs.get('activation_type', 'relu')
        hsz = int(kwargs['hsz'])
        return WithoutLength(ConvEncoderStack(input_dim, hsz, filtsz, layers, pdrop, activation_type))


@register_model(task='tagger', name='pass')
class PassThruTaggerModel(AbstractEncoderTaggerModel):
    """A Pass-thru implementation of the encoder

    When we fine-tune our taggers from things like BERT embeddings, we might want to just pass through our
    embedding result directly to the output decoder.  This model provides a mechanism for this by providing
    a simple identity layer
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, input_dim: int, **kwargs) -> BaseLayer:
        """Identity layer encoder

        :param input_dim: The input dims
        :param kwargs: None
        :return: An encoder
        """
        return WithoutLength(PassThru(input_dim))
