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
        super(AbstractArcPolicy, self).__init__(pc)

    def get_state(self, encoder_outputs):
        pass

    def forward(self, encoder_output, hsz, bead_width=1):
        pass


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(ArcPolicy):
    def __init__(self, pc):
        super(TransferLastHiddenPolicy, self).__init__(pc)

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_arc_policy(name='no_arc')
class NoArcPolicy(ArcPolicy):
    def __init__(self, pc):
        super(NoArchPolicy, self).__init__(pc)

    def get_state(self, encoder_outputs):
        final_state = encoder_outputs.hidden
        return 'BUTTS'
