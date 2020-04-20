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
    def create(cls, embeddings, labels, **kwargs):

        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.5)
        model.lengths_key = kwargs.get('lengths_key')
        model.embeddings = embeddings
        embed_model = model.init_embed(**kwargs)
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels
        pool_model = model.init_pool(embed_model.dsz, **kwargs)
        stack_model = model.init_stacked(pool_model.output_dim, **kwargs)
        output_model = model.init_output(stack_model.output_dim if stack_model else pool_model.output_dim, len(labels), **kwargs)
        model.layers = EmbedPoolStackModel(len(labels), embed_model, pool_model, stack_model, output_model)
        logger.info(model)
        return model

    def cuda(self, device=None):
        super().cuda(device=device)
        self.gpu = True

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

    def forward(self, input: Dict[str, torch.Tensor]):
        return self.layers(input)

    def predict_batch(self, batch_dict, **kwargs):
        numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
        examples, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        with torch.no_grad():
            probs = self(examples).exp()
            probs = unsort_batch(probs, perm_idx)
        return probs

    def predict(self, batch_dict, raw=False, dense=False, **kwargs):
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

    def get_labels(self):
        return self.labels

    def init_embed(self, **kwargs):
        """Produce the embeddings operation that will be used in the model

        :param kwargs:
        :return: An embeddings operation
        """
        return EmbeddingsStack(self.embeddings, reduction=kwargs.get('embeddings_reduction', 'concat'))

    def init_pool(self, dsz, **kwargs):
        """Produce a pooling operation that will be used in the model

        :param dsz: The input dimension size
        :param kwargs:
        :return: A pooling operation
        """
        pass

    def init_stacked(self, input_dim, **kwargs):
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            return None
        return DenseStack(input_dim, hszs)

    def init_output(self, input_dim, output_dim, **kwargs):
        if input_dim is None:
            return None
        return Dense(input_dim, output_dim, activation=kwargs.get('output_activation', 'log_softmax'))


@register_model(task='classify', name='default')
class ConvModel(ClassifierModelBase):

    def init_pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        return WithoutLength(WithDropout(ParallelConv(dsz, cmotsz, filtsz, "relu", input_fmt="bth"), self.pdrop))


@register_model(task='classify', name='lstm')
class LSTMModel(ClassifierModelBase):

    def init_pool(self, dsz, **kwargs):
        unif = kwargs.get('unif')
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        weight_init = kwargs.get('weight_init', 'uniform')
        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        if rnntype == 'blstm':
            return BiLSTMEncoderHidden(dsz, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)
        return LSTMEncoderHidden(dsz, hsz, 1, self.pdrop, unif=unif, batch_first=True, initializer=weight_init)

    def make_input(self, batch_dict):
        inputs = super().make_input(batch_dict)
        lengths = inputs['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        for k, value in inputs.items():
            inputs[k] = value[perm_idx]
        return inputs


class NBowModelBase(ClassifierModelBase):
    """This base classes forces at least one stacked layer for NBow models"""

    def init_stacked(self, input_dim, **kwargs):
        """Produce a stacking operation that will be used in the model


        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        kwargs.setdefault('hsz', [100])
        return super().init_stacked(input_dim, **kwargs)


@register_model(task='classify', name='nbow')
class NBowModel(NBowModelBase):

    def init_pool(self, dsz, **kwargs):
        return MeanPool1D(dsz)


@register_model(task='classify', name='nbowmax')
class NBowMaxModel(NBowModelBase):

    def init_pool(self, dsz, **kwargs):
        return MaxPool1D(dsz)


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
        inputs = super(CompositePoolingModel, self).make_input(batch_dict)
        lengths = inputs['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        for k, value in inputs.items():
            inputs[k] = value[perm_idx]
        return inputs


@register_model(task='classify', name='fine-tune')
class FineTuneBaselineModel(ClassifierModelBase):
    """Fine-tune based on pre-pooled representations"""

    @classmethod
    def create(cls, embeddings, labels, **kwargs):

        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.5)
        model.lengths_key = kwargs.get('lengths_key')
        model.embeddings = embeddings
        embed_model = model.init_embed(**kwargs)
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels
        stack_model = model.init_stacked(embed_model.output_dim, **kwargs)
        model.layers = FineTuneModel(len(labels), embeddings, stack_model)
        logger.info(model)
        return model
