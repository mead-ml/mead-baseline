import os
import logging
from baseline.utils import write_json
from baseline.pytorch.torchy import *
from baseline.pytorch.transformer import *
from baseline.model import EncoderDecoderModel, register_model, create_seq2seq_encoder, create_seq2seq_decoder
from baseline.pytorch.seq2seq.encoders import *
from baseline.pytorch.seq2seq.decoders import *

logger = logging.getLogger('baseline')


class EncoderDecoderModelBase(nn.Module, EncoderDecoderModel):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super().__init__()
        self.beam_sz = kwargs.get('beam', 1)
        self.gpu = kwargs.get('gpu', True)
        src_dsz = self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.dropin_values = kwargs.get('dropin', {})
        self.encoder = self.init_encoder(src_dsz, **kwargs)
        self.decoder = self.init_decoder(tgt_embedding, **kwargs)

    def init_embed(self, src_embeddings, tgt_embedding, **kwargs):
        """This is the hook for providing embeddings.  It takes in a dictionary of `src_embeddings` and a single
        tgt_embedding` of type `PyTorchEmbedding`

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings, one per embedding
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        :return: Return the aggregate embedding input size
        """
        self.src_embeddings = EmbeddingsStack(src_embeddings, reduction=kwargs.get('embeddings_reduction', 'concat'))
        return self.src_embeddings.output_dim

    def init_encoder(self, input_sz, **kwargs):
        # This is a hack since TF never needs this one, there is not a general constructor param, so shoehorn
        kwargs['dsz'] = input_sz
        return create_seq2seq_encoder(**kwargs)

    def init_decoder(self, tgt_embedding, **kwargs):
        return create_seq2seq_decoder(tgt_embedding, **kwargs)

    def encode(self, input, lengths):
        """

        :param input:
        :param lengths:
        :return:
        """
        embed_in_seq = self.embed(input)
        return self.encoder(embed_in_seq, lengths)

    def decode(self, encoder_outputs, dst):
        return self.decoder(encoder_outputs, dst)

    def save(self, model_file):
        """Save the model out

        :param model_file: (``str``) The filename
        :return:
        """
        torch.save(self, model_file)

    def create_loss(self, **kwargs):
        """Create a loss function.

        :return:
        """
        label_smoothing = kwargs.get("label_smoothing")
        if label_smoothing is not None:
            def make(*args, **kwargs):
                return LabelSmoothingLoss(label_smoothing, *args, **kwargs)
            return SequenceLoss(make)
        return SequenceLoss()

    @classmethod
    def load(cls, filename, **kwargs):
        """Load a model from file

        :param filename: (``str``) The filename
        :param kwargs:
        :return:
        """
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        model = cls(src_embeddings, tgt_embedding, **kwargs)
        logger.info(model)
        return model

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)

        if not self.training or v == 0:
            return x

        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx, numpy_to_tensor=False):
        tensor = batch_dict[key]
        if numpy_to_tensor:
            tensor = torch.from_numpy(tensor)

        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):
        """Prepare the input.

        :param batch_dict: `dict`: The data.
        :param perm: `bool`: If True return the permutation index
            so that you can undo the sort if you want.
        """
        example = dict({})

        lengths = batch_dict[self.src_lengths_key]
        if numpy_to_tensor:
            lengths = torch.from_numpy(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)

        #if self.gpu:
        #    lengths = lengths.cuda()
        example['src_len'] = lengths
        for key in self.src_embeddings.keys():
            example[key] = self.input_tensor(key, batch_dict, perm_idx, numpy_to_tensor=numpy_to_tensor)

        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt']
            if numpy_to_tensor:
                tgt = torch.from_numpy(tgt)
            example['dst'] = tgt[:, :-1]
            example['tgt'] = tgt[:, 1:]
            example['dst'] = example['dst'][perm_idx]
            example['tgt'] = example['tgt'][perm_idx]
            if self.gpu:
                example['dst'] = example['dst'].cuda()
                example['tgt'] = example['tgt'].cuda()
        if perm:
            return example, perm_idx
        return example

    def embed(self, input):
        return self.src_embeddings(input)

    def forward(self, input: Dict[str, torch.Tensor]):
        src_len = input['src_len']
        encoder_outputs = self.encode(input, src_len)
        output = self.decode(encoder_outputs, input['dst'])
        # Return as B x T x H
        return output

    def predict(self, batch_dict, **kwargs):
        """Predict based on the batch.

        If `make_input` is True then run make_input on the batch_dict.
        This is false for being used during dev eval where the inputs
        are already transformed.
        """
        self.eval()
        make = kwargs.get('make_input', True)
        if make:
            numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
            inputs, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        else:
            inputs = batch_dict
        encoder_outputs = self.encode(inputs, inputs['src_len'])
        outs, lengths, scores = self.decoder.beam_search(encoder_outputs, **kwargs)
        if make:
            outs = unsort_batch(outs, perm_idx)
            lengths = unsort_batch(lengths, perm_idx)
            scores = unsort_batch(scores, perm_idx)
        return outs.cpu().numpy(), scores.cpu().numpy()


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2SeqModel(EncoderDecoderModelBase):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        """This base model is extensible for attention and other uses.  It declares minimal fields allowing the
        subclass to take over most of the duties for drastically different implementations

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        """
        super().__init__(src_embeddings, tgt_embedding, **kwargs)
