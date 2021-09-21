from baseline.pytorch.torchy import *
from baseline.model import register_model
from baseline.pytorch.lm.model import TransformerLanguageModel
from eight_mile.pytorch.serialize import load_tlm_npz
import torch.autograd
import os

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