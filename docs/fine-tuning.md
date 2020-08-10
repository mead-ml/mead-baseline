# Fine-Tuning, Pre-Training and Transfer Learning in MEAD

MEAD has extensive support for fine-tuning and transfer learning and language moodel-base self-supervised pre-training.
Support for Transformer-based pre-training is built into the core library, and you can find examples of this in
the [sample mead config files](../mead/config) or in the [mead-tutorials](https://github.com/dpressel/mead-tutorials)

Fine-tuning models are typically applied as an Embedding model in MEAD which contains the entire sub-graph of the upstream model
except for its training head.  For simple fine-tuning of classifiers, use a `FineTuneModel` to add a classification head.
For simple fine-tuning of taggers, you can use a `PassThruModel` model to add a single tagger head, and there are
examples of how to configure these in the [sample mead configs](../mead/config)
 
For additional embeddings for fine-tuning, see also the embeddings provided at [mead-hub](https://github.com/mead-ml/hub)

## Fine-Tuning Transformers

#### Using the Library

We provide Transformer contextual Embeddings built on our core layers (called 8-mile), which supports a broad variety of
Transformers.  For example, BERT, a bidirectionally trained transformer can be loaded into these Embeddings and trained as part of a downstream task.

There are several common Transformer variants.  The original Transformer paper places `LayerNorm` layers
after each transform, and uses sinusoidal (fixed) positional embeddings, but subsequent research has found that
optimal placement is at the beginning of each block.

- *Embeddings* - Most versions of Transformers since GPT use learned positional embeddings, rather than using the
sinusoidal embeddings.  If sinusoidal embeddings are used, there are differences between the definition in the paper and
most implementations.

- *LayerNorm positioning* - Most recent implementations place the `LayerNorm` before any of the `Transformer` block
operations, which is the default in `8-mile`.  To support either variant, we offer a boolean to tell the 
blocks where to put the transform.

- *Relative position representations* - The [Shaw et al., 2018](https://arxiv.org/pdf/1803.02155.pdf) paper defines a modification to Multi-headed attention to support
relative positional embeddings.  This work demonstrates that RA is often more effective than positional global embeddings,
and can be used in place of them.  Following this, models may define the usual `LookupTableEmbeddings` instead of the
positional flavors

- *BERT flavored Transformer*:  BERT follows the original T2T implementation but substitutes learned positional embeddings
instead of sinusoidal ones.  To set up your MEAD config to construct the proper Transformer model, the config `features`
block should look like this:
follows:

```
 features:
- embeddings:
    word_embed_type: learned-positional-w-bias # See the notes below for an explanation of this embedding type
    label: bert-base-uncased-npz               # A label defined in the embeddings index
    type: tlm-words-embed-pooled               # The embedding class we are using
    reduction: sum-layer-norm                  # BERT does layer norm after embeddings before Transformer encoding
    layer_norms_after: True                    # BERT does layer norm after the Transformer block not before it
    finetune: True                             # Don't freeze the weights!
    dropout: 0.1                               # Dropout on the BERT sub-graph
    mlm: True                                  # This specifies the type of mask to apply (vs causual mask)
  name: bert
  vectorizer:
    label: bert-base-uncased
```

The model block for fine-tuning has the name `fine-tune`:



#### Fine-Tuning HuggingFace Transformers

The developers at HuggingFace have been providing faithful PyTorch implementations of nearly all of the popular 
Transformer models produced by the research community, and have carefully validated each model matches the original
implementations exactly.

While we provide native support for Transformer models, if you prefer or need to use the HuggingFace libraries to fine-tune models in MEAD, its fairly easy to do [in an addon](https://github.com/mead-ml/hub/blob/master/v1/addons/embed_bert_pytorch.py).

#### Fine-Tuning BERT Official

We provide an [addon on mead-hub](https://github.com/mead-ml/hub/blob/master/v1/addons/embed_bert_tf.py) to use either the BERT official code (a copy) or the TF Hub module to fine-tune.
Since this is based on the official code, no conversion or non-standard checkpoints are required.


## Pre-Training your own Transformers with MEAD

### Using the API Examples

MEAD has a rich set of pre-training solutions provided in the API examples, with support for a wide-variety of
Transformer-based pre-training.  For pre-training, which often requires training multi-GPU, multi-worker, we support 
PyTorch DistributedDataParallel facilities to train and TF2.0 autofunction-based distribution and each program exports to a
framework-independent NPZ file that can be loaded by either TensorFlow or PyTorch MEAD layers.


#### Pre-Training Transformer-based Language Models with MEAD

By now, the practice of pre-training Transformer models on a large amount of data to be used for downstream fine-tuning
is ubiquitous.  MEAD supports pre-training architectures such as BERT/RoBERTa via several scripts, all of which
support multi-worker training.

* [pretrain_tlm_pytorch](../api-examples/pretrain_tlm_pytorch.py): Train with multi-worker configuration with a large
(possibly out-of-core) LM dataset, using either `fastBPE` or `WordPiece` to tokenize the words (PyTorch).
This program supports export of torch checkpoints as well as NPZ checkpoints, and supports reloading from either.
Supports Kubernetes-based training with PyTorchJob operator

* [pretrain_tlm_tf](../api-examples/pretrain_tlm_tf.py): Train multi-worker on TPUs or GPUs with a large
(possibly out-of-core) LM dataset, using [fastBPE](https://github.com/glample/fastBPE) to tokenize the words.  This program supports export of TF checkpoints
as well as NPZ checkpoints.  Supports training on TPUs as well as Kubernetes-based training with either TFJob operator, Job or Pod CRDs.
  - There is a [stripped down example Google Colab tutorial](https://colab.research.google.com/github/dpressel/mead-tutorials/blob/master/mead_transformers_tpu.ipynb) based on this program where you can train an MLM on a TPU

* [pretrain_paired_pytorch](../api-examples/pretrain_paired_pytorch.py): Train either a Transformer encoder-decoder or a Transformer
dual encoder out-of-core on very large datasets.  The dual-encoder follows [ConveRT, Henderson et al 2019](https://arxiv.org/pdf/1911.03688.pdf).
This variant creates 2 context, one for the encoder and one for the
decoder, either based on splitting a single line in 2 (Next Sequence Prediction or NSP), or by separating the line 
by a tab delimiter (Next Turn Prediction or NTP).  In the latter case, its presumed that the separator will partition
the line into a query and a response.

* [pretrain_discrim](../api-examples/pretrain_discrim.py): In this approach, we jointly train a generator and a
discriminator following [ELECTRA: Pre-Training Text Encoders as Discriminators Rather Than Generators,
    Clark et al. 2019](https://openreview.net/pdf?id=r1xMH1BtvB).

#### Writing your own Pre-Training script

Its pretty easy to tweak the above examples to create your own new training regime with Transformers.  Copy one
of the other training examples that is closest to what you are trying to do, and modify the `create_loss()` and or
the model architecture accordingly, and kick it off on your machine (or a Kubernetes cluster) and make sure you
follow the structure of the other examples.  If you can use the existing baseline models 
`TransformerMaskedLanguageModel` or `TransformerLanguageModel`, this will make things the easiest.  If not, you can
create a similar class to one of those and change the portions you need (or subclass them).

- If you need to create your own Transformer LM
  - make sure that the internal structure of the encoder is named either `generator` or `transformer` so that the 
  serializers/converters know where to find them.
  - make sure to use an `EmbeddingsStack` named `embeddings` to contain your embeddings objects, and use some form of
  `*PositionalEmbeddingsModel` or `LookupTableEmbeddingsModel` for your embeddings where possible to avoid having to
  create custom serializers/converters

### Testing out your Pre-Trained model

If you have pre-trained an MLM (like BERT) or an Encoder-Decoder (like T5), you might want to see what
kinds of reconstructions it makes given some data.  For MLMs, use [generate_mlm](../api-examples/generate_mlm.py) to
test out the model on various masked utterances:

```
$ python generate_mlm.py --sample true --rpr_k 48 --nctx 128 --subword_model_file codes.30k --subword_vocab_file vocab.30k --query " which do you <unk> , coke or <unk> . i prefer <unk> !" --checkpoint checkpoint-step-1208324.npz
[Query] which do you <unk> , coke or <unk> . i prefer <unk> !
[BPE] which do you [MASK] , coke or [MASK] . i prefer [MASK] ! <EOU>
[Response] which do you prefer , coke or pepsi . i prefer coke ! <EOU>

```

To explore results from an encoder-decoder:

```
$ python transformer_seq2seq_response.py --subword_model_file codes.30k --subword_vocab_file vocab.30k --checkpoint checkpoint-step-483329.npz --device cpu --query "so <unk> vs coke what do you <unk> ." --sample true --go_token "<PAD>"

[Query] so <unk> vs coke what do you <unk> . <EOU>
[Response] so pepsi vs coke what do you think . <EOU>


```

### Transformer Checkpoint conversion

The checkpoints are written to NPZ format using the 8-mile library, which writes each layer from its native format into
an NPZ and can hydrate a model from those checkpoints as well.  The checkpoints are roughly the same size as a PyTorch
checkpoint.

Please note that the checkpoints that are written during pre-training or export are headless as they are typically used
for downstream processing.  However, for models with tied embeddings (most models), the heads will automatically be
restored to the tied module since `weight` is property that is tied to the other module's `.weight`


