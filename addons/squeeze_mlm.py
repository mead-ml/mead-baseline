from baseline.pytorch.torchy import *
from baseline.model import register_model
from baseline.pytorch.lm.model import TransformerLanguageModel
from eight_mile.pytorch.serialize import load_tlm_npz
import torch.autograd
import os


class PooledSequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.BCEWithLogitsLoss, avg='token'):
        super().__init__()
        if avg == 'token':
            self.crit = LossFn()
            self._norm = self._no_norm
        else:
            self.crit = LossFn()
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        #inputs = inputs.transpose(0, 1)
        C = inputs.shape[-1]
        flat_targets = torch.nn.functional.one_hot(targets, C)

        # Get the offsets of the non-zero targets, the values of these are all on
        flat_targets = (torch.sum(flat_targets, axis=1) != 0).float()
        flat_targets[:, Offsets.PAD] = 0
        flat_targets[:, Offsets.EOS] = 0
        flat_targets[:, Offsets.GO] = 0

        if len(inputs.shape) > 2:
            max_per_vocab = inputs.max(0)[0]
            loss = self.crit(max_per_vocab, flat_targets)
        else:
            loss = self.crit(inputs, flat_targets)
        return self._norm(loss, inputs)

@register_model(task='lm', name='squeeze-freeze-mlm')
class SqueezeFreezeMaskedLanguageModel(TransformerLanguageModel):

    def init_output(self, embeddings, **kwargs):
        self.vsz = embeddings[self.tgt_key].get_vsz()
        hsz = kwargs.get('hsz', kwargs.get('d_model'))

        unif = float(kwargs.get('unif', 0.0))
        output = pytorch_linear(hsz*2, self.vsz, unif)
        return output

    def create_layers(self, embeddings, **kwargs):
        super().create_layers(embeddings, **kwargs)
        self.mask_pad = True

    def create_mask(self, bth, inputs):
        if not self.mask_pad:
            return None

        return self._pad_mask(inputs)

    def generate(self, bth, _, inputs):
        with torch.no_grad():
            mask = self.create_mask(bth, inputs)
            xt = bth
            for i in range(len(self.generator.encoders)-1):
                layer = self.generator.encoders[i]
                xt = layer((xt, mask))

        xt = self.generator.encoders[-1]((xt, mask))
        xt = self.generator.ln(xt)
        B = xt.shape[0]
        zero = xt[:, 0]
        lengths = mask.squeeze(1).squeeze(1).sum(-1)
        last = xt[torch.arange(B), lengths - 1]
        pooled = torch.cat([zero, last], -1)
        return pooled, None

    def create_loss(self):
        return PooledSequenceCriterion()#LossFn=nn.CrossEntropyLoss)

    def forward(self, input: Dict[str, TensorDef], hidden: TensorDef) -> Tuple[TensorDef, TensorDef]:
        with torch.no_grad():
            emb = self.embed(input)
        output, hidden = self.generate(emb, hidden, input)
        return self.output_layer(output), hidden


@register_model(task='lm', name='squeeze-mlm')
class SqueezeMaskedLanguageModel(TransformerLanguageModel):

    def init_output(self, embeddings, **kwargs):
        self.vsz = embeddings[self.tgt_key].get_vsz()
        hsz = kwargs.get('hsz', kwargs.get('d_model'))

        unif = float(kwargs.get('unif', 0.0))
        output = pytorch_linear(hsz*2, self.vsz, unif)
        return output

    def create_layers(self, embeddings, **kwargs):
        super().create_layers(embeddings, **kwargs)
        self.freeze_encoder = kwargs.get('freeze_encoder', True)
        self.reduction_layer = TwoHeadConcat(kwargs.get('hsz'), dropout=kwargs.get('dropout', 0.1), pooling="mean")

    def create_mask(self, bth, inputs):
        if not self.mask_pad:
            return None

        return self._pad_mask(inputs)

    def generate(self, bth, _, inputs):
        with torch.no_grad() if self.freeze_encoder else contextlib.ExitStack():
            mask = self.create_mask(bth, inputs)
            transduce = self.generator((bth, mask))
        if mask is None:
            mask = self._pad_mask(inputs)
        pooled = self.reduction_layer((transduce, transduce, transduce, mask))
        return pooled, None

    def create_loss(self):
        return PooledSequenceCriterion()#LossFn=nn.CrossEntropyLoss)

    def forward(self, input: Dict[str, TensorDef], hidden: TensorDef) -> Tuple[TensorDef, TensorDef]:
        with torch.no_grad() if self.freeze_encoder else contextlib.ExitStack():
            emb = self.embed(input)
        output, hidden = self.generate(emb, hidden, input)
        return self.output_layer(output), hidden