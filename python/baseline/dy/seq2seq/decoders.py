

class ArcPolicy(DynetModel):
    def __init__(self, pc):
        super(ArchPolicy, self).__init__(pc)

    def forward(self, encoder_outputs, hsz, beam_width=1)
