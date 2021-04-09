"""
Load a pretrained Transformer Tagger

Its pretrained with TensorFlow using TPUs.  Here is the model definition:

```

class TransformerTagger(tf.keras.Model):
    def __init__(
        self,
        num_labels: int,
        embeddings: tf.keras.layers.Layer,
        num_heads=8, num_layers=8, d_model=512, d_ff=None, rpr_k=None, activation='gelu', scale=True, ffn_pdrop=0.0,
        layer_norm_eps=1.0e-6, rpr_value_on=True, layer_drop=0.0, dropout=0.1,
        name: str = None, **kwargs

    ):
        super().__init__(name=name)
        self.embeddings = EmbeddingsStack(embeddings)
        self.transformer = TransformerEncoderStack(
            num_heads, d_model=d_model,
            pdrop=dropout, scale=scale,
            layers=num_layers, d_ff=d_ff, rpr_k=rpr_k,
            activation=activation,
            ffn_pdrop=ffn_pdrop,
            layer_norm_eps=layer_norm_eps,
            rpr_value_on=rpr_value_on,
            layer_drop=layer_drop)
        self.output_layer = tf.keras.layers.Dense(num_labels)

    def call(self, inputs):
        input_mask = tf.expand_dims(tf.expand_dims(tf.cast(inputs['x'] != Offsets.PAD, tf.int32), 1), 1)
        embed = self.embeddings(inputs)
        transformed = self.transformer((embed, input_mask))
        output = self.output_layer(transformed)
        return output
```

"""
from eight_mile.utils import read_json
from eight_mile.pytorch.layers import *
from eight_mile.pytorch.serialize import load_tlm_output_npz
from baseline.pytorch.tagger import TaggerModelBase, register_model
from baseline.pytorch.torchy import TensorDef, BaseLayer


@register_model(task='tagger', name='pretrained')
class TransformerTagger(TaggerModelBase):
    """Transformer tagger impl. matching the version saved in TensorFlow

    To reuse the output layer as expected, its critical that you define the full label list in your dataset entry.

    This is easy to do, just copy the labels.json from pretraining into the YAML or json file under a `label_list`
    field:

    ```
    - label: conll-iobes
      train_file: /data/datasets/ner/conll-iobes/eng.train.iobes
      valid_file: /data/datasets/ner/conll-iobes/eng.testa.iobes
      test_file: /data/datasets/ner/conll-iobes/eng.testb.iobes
      label_list: {
        "<PAD>": 0,
        "<GO>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "B-PER": 4,
        "I-PER": 5,
        "E-PER": 6,
        "S-PER": 7,
        "B-LOC": 8,
        "I-LOC": 9,
        "E-LOC": 10,
        "S-LOC": 11,
        "B-ORG": 12,
        "I-ORG": 13,
        "E-ORG": 14,
        "S-ORG": 15,
        "B-MISC": 16,
        "I-MISC": 17,
        "E-MISC": 18,
        "S-MISC": 19,
        "O": 20
      }
    ```

    """
    def create_layers(self, embeddings: Dict[str, TensorDef],
                      num_heads=8, num_layers=8, d_model=512, d_ff=None, rpr_k=None, activation='gelu', scale=True,
                      ffn_pdrop=0.0,
                      layer_norm_eps=1.0e-6, rpr_value_on=True, layer_drop=0.0, dropout=0.1,
                      name: str = None,
                      checkpoint: str = None,
                      **kwargs):


        # We are only going to allow one embedding for now, and we want it to be renamed to 'x'
        self.key = list(embeddings.keys())[0]
        self.embeddings = EmbeddingsStack({'x': embeddings[self.key]})
        self.transformer = TransformerEncoderStack(
            num_heads, d_model=d_model,
            pdrop=dropout, scale=scale,
            layers=num_layers, d_ff=d_ff, rpr_k=rpr_k,
            activation=activation,
            ffn_pdrop=ffn_pdrop,
            layer_norm_eps=layer_norm_eps,
            rpr_value_on=rpr_value_on,
            layer_drop=layer_drop)
        self.output = Dense(d_model, len(self.labels))

        constraint_mask = kwargs.get('constraint_mask')
        if constraint_mask is not None:
            constraint_mask = constraint_mask.unsqueeze(0)

        self.decoder = TaggerGreedyDecoder(
            len(self.labels),
            constraint_mask=constraint_mask,
            batch_first=True,
            reduction=kwargs.get('reduction', 'batch')
        )

        logger.info(self)

        if checkpoint is not None:
            if os.path.isdir(checkpoint):
                checkpoint, _ = find_latest_checkpoint(checkpoint)
            logger.info("Reloading checkpoint [%s]", checkpoint)
            # Use this if you want to verify that your labels are all the same.  If not, something bad is going to
            # happen
            # To ensure this, make a dataset index with a proper label set defined:

            #labels = read_json(os.path.join(os.path.dirname(checkpoint), 'labels.json'))

            if checkpoint.endswith(".npz"):

                load_tlm_output_npz(self, checkpoint)
            else:
                raise Exception("Currently we only support NPZ format checkpoints")
        else:
            logger.warning("No checkpoint was given.  Randomly initializing the weights")

    def transduce(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """This operation performs embedding of the input, followed by encoding and projection to logits

        :param inputs: The feature indices to embed
        :return: Transduced (post-encoding) output
        """
        lengths = inputs['lengths']
        x = inputs[self.key]
        inputs = {'x': x, 'lengths': lengths}

        embedded = self.embeddings(inputs)
        mask = sequence_mask(lengths, x.shape[1]).unsqueeze(1).unsqueeze(1).to(device=embedded.device)
        transformer_output = self.transformer((embedded, mask))
        transduced = self.output(transformer_output)
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

    @classmethod
    def load(cls, filename: str, **kwargs) -> 'TaggerModelBase':
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model
