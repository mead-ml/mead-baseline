import logging
from baseline.model import DependencyParserModel, register_model
from baseline.pytorch.torchy import *
from eight_mile.pytorch.layers import sequence_mask_mxlen, truncate_mask_over_time
from baseline.utils import listify, write_json, revlut
from eight_mile.pytorch.layers import *
import torch.backends.cudnn as cudnn
import torch.jit as jit
import os
cudnn.benchmark = True

logger = logging.getLogger('baseline')


def decode_results(heads_pred: TensorDef, labels_pred: TensorDef) -> Tuple[TensorDef, TensorDef]:
    # Just do a quick greedy decode, pick the argmax of the heads, and for that head, pick the
    # argmax of the label
    B = labels_pred.shape[0]
    T = labels_pred.shape[1]

    # If there is padding, rip it off to the max sequence length so the tensors are the same size
    greedy_heads_pred = torch.argmax(heads_pred, -1).view(-1)
    greedy_labels_pred = labels_pred.reshape(B * T, T, -1)[
        torch.arange(len(greedy_heads_pred)), greedy_heads_pred].view(B, T, -1)

    greedy_labels_pred = torch.argmax(greedy_labels_pred, -1)
    greedy_heads_pred = greedy_heads_pred.view(B, T)
    return greedy_heads_pred, greedy_labels_pred


class ArcLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.arc_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.label_loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, arcs_pred, arcs_gold, labels_pred, labels_gold):
        # First, trim gold labels to length of the input
        B = labels_pred.shape[0]
        T = labels_pred.shape[1]
        flat_labels_pred = labels_pred.reshape(B*T, T, -1)
        flat_arcs_pred = arcs_pred.reshape(B*T, T)
        flat_labels_gold = labels_gold[:, :T].contiguous().view(-1)
        flat_arcs_gold = arcs_gold[:, :T].contiguous().view(-1)
        flat_labels_pred = flat_labels_pred[torch.arange(len(flat_arcs_gold)), flat_arcs_gold]
        arc_loss = self.arc_loss(flat_arcs_pred, flat_arcs_gold)
        rel_loss = self.label_loss(flat_labels_pred, flat_labels_gold)
        return arc_loss + rel_loss


class DependencyParserModelBase(nn.Module, DependencyParserModel):

    def __init__(self):
        super().__init__()
        self.gpu = False

    @classmethod
    def load(cls, filename: str, **kwargs) -> 'DependencyParserModelBase':
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
    def create(cls, embeddings, labels, **kwargs) -> 'DependencyParserModelBase':

        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.333)
        model.dropin_values = kwargs.get('dropin', {})
        model.lengths_key = kwargs.get('lengths_key')
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels["labels"]
        model.punct = labels["labels"].get("punct", Offsets.PAD)
        model.create_layers(embeddings, **kwargs)
        logger.info(model)
        return model

    def cuda(self, device=None):
        self.gpu = True
        return super().cuda(device=device)

    def create_loss(self):
        return ArcLabelLoss()

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
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx, numpy_to_tensor=numpy_to_tensor)

        y = batch_dict.get('heads')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)

            if perm_idx is not None:
                y = y[perm_idx]
            if self.gpu:
                y = y.cuda()
            example_dict['heads'] = y

        y = batch_dict.get('labels')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if perm_idx is not None:
                y = y[perm_idx]
            if self.gpu:
                y = y.cuda()
            example_dict['labels'] = y

        if perm:
            return example_dict, perm_idx

        return example_dict

    def predict_batch(self, batch_dict: Dict[str, TensorDef], **kwargs) -> TensorDef:
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        decode = bool(kwargs.get('decode', True))
        examples, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        with torch.no_grad():
            if decode:
                arcs, rels = self.decode(examples)

            else:
                arcs, rels = self(examples)
                arcs.exp_()
                rels.exp_()
            arcs = unsort_batch(arcs, perm_idx)
            rels = unsort_batch(rels, perm_idx)

        return arcs, rels

    def predict(self, batch_dict: Dict[str, TensorDef], **kwargs):
        """Raw prediction, for non-fancy decoding, use `def decode()` instead

        :param batch_dict:
        :param kwargs:
        :return:
        """
        self.eval()
        arcs, rels = self.predict_batch(batch_dict, **kwargs)
        return arcs, rels

    def get_labels(self) -> List[str]:
        return self.labels

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def decode(self, example, **kwargs):
        heads_pred, labels_pred = self(example)
        return decode_results(heads_pred, labels_pred)


@register_model(task='deps', name='default')
class BiAffineDependencyParser(DependencyParserModelBase):

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        output_dim_arcs = kwargs.get('hsz_arcs', 500)
        self.arc_h = self.init_proj(self.pool_model.output_dim, output_dim_arcs, **kwargs)
        self.arc_d = self.init_proj(self.pool_model.output_dim, output_dim_arcs, **kwargs)
        output_dim_rels = kwargs.get('hsz_rels', 100)
        self.rel_h = self.init_proj(self.pool_model.output_dim, output_dim_rels, **kwargs)
        self.rel_d = self.init_proj(self.pool_model.output_dim, output_dim_rels, **kwargs)
        self.arc_attn = self.init_biaffine(self.arc_h.output_dim, 1, True, False)
        self.rel_attn = self.init_biaffine(self.rel_h.output_dim, len(self.labels), True, True)
        self.primary_key = self.lengths_key.split('_')[0]

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
        embeddings_dropout = float(kwargs.get('embeddings_dropout', self.pdrop))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a pooling operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A pooling operation
        """
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 800))
        layers = kwargs.get('layers', 3)
        if layers == 0:
            logger.warning("No layers given. Setting up pass-through model")
            return WithoutLength(PassThru(input_dim))
        return BiLSTMEncoderSequence(input_dim, hsz, layers, self.pdrop, batch_first=True, initializer="ortho")


    def init_proj(self, input_dim: int, output_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        return WithDropout(Dense(input_dim, output_dim, activation=kwargs.get('activation', 'leaky_relu'), initializer="ortho"),
                           pdrop=self.pdrop)

    def init_biaffine(self, input_dim: int, output_dim: int, bias_x: bool, bias_y: bool):
        return BilinearAttention(input_dim, output_dim, bias_x, bias_y)



    def forward(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """

        lengths = inputs.get("lengths")
        Tin = inputs[self.primary_key].shape[1]
        mask = sequence_mask_mxlen(lengths, max_len=Tin).to(lengths.device)
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        arcs_h = self.arc_h(pooled)
        arcs_d = self.arc_d(pooled)
        rels_h = self.rel_h(pooled)
        rels_d = self.rel_d(pooled)
        mask = truncate_mask_over_time(mask, arcs_h)
        score_arcs = self.arc_attn(arcs_d, arcs_h, mask)
        score_rels = self.rel_attn(rels_d, rels_h, mask.unsqueeze(1)).permute(0, 2, 3, 1)
        return score_arcs, score_rels
