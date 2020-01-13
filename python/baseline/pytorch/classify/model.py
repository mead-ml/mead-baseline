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
        m = torch.jit.script(self)
        m.save(f'{outname}.script')
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
        model.layers = EmbedPoolStackModel(len(labels), embed_model, pool_model, stack_model)
        logger.info(model)
        return model

    def cuda(self, device=None):
        super().cuda(device=device)
        self.gpu = True

    def create_loss(self):
        return nn.NLLLoss()

    def make_input(self, batch_dict):
        """Transform a `batch_dict` into something usable in this model

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :return:
        """
        example_dict = dict({})
        for key in self.embeddings.keys():
            example_dict[key] = torch.from_numpy(batch_dict[key])
            if self.gpu:
                example_dict[key] = example_dict[key].cuda()

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            example_dict['lengths'] = torch.from_numpy(batch_dict[self.lengths_key])
            if self.gpu:
                example_dict['lengths'] = example_dict['lengths'].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        return example_dict

    def forward(self, input: Dict[str, torch.Tensor]):
        return self.layers(input)

    def predict(self, batch_dict):
        examples = self.make_input(batch_dict)

        with torch.no_grad():
            probs = self(examples).exp()
            probs.div_(torch.sum(probs))
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
        return EmbeddingsStack(self.embeddings)

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
