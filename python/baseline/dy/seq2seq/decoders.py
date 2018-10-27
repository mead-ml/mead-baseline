import dynet as dy
from baseline.utils import Offsets
from baseline.dy.transformer import subsequent_mask, TransformerDecoderStack
from baseline.model import (
    register_decoder,
    register_arc_policy,
    create_seq2seq_arc_policy
)

class ArcPolicy(DynetModel):
    def __init__(self, pc):
        super(ArcPolicy, self).__init__(pc)

    def forward(self, encoder_outputs, hsz, beam_width=1)
