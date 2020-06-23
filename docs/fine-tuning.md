# Fine-tuning, Pre-Training and Transfer Learning in MEAD

MEAD has supported fine-tuning and transfer learning since its initial release.  The typical recipe for MEAD is
to define an `*EmbeddingsModel` that can load a pretrained checkpoint inside of an `addon`, which is simply a
custom extension to MEAD with a `@register_embeddings` directive.

In many cases, the user wants to try out some new model where the checkpoint has been posted online, and just
encode the words in a sequence with that model, and then either fine-tune (by adding a single layer to the logits/labels
output) or by using a previously defined model architecture with a custom embedding.

For most cases, this is quite simple.  We provide embeddings for several well-known models including *BERT*
and *ELMo*.


## Fine-tuning Transformers

#### Natively

We provide a Transformer-embedding built on our core layers (called 8-mile), which supports a broad variety of
Transformers.

Since BERT is actually just a normal transformer with an MLM objective
(and possibly NSP objective) and since MEAD has built-in layers for building your own Transformer,
BERT is actually a special case where no `addon` is required to load the model at all and the model can be fine-tuned
with only the core library.

It is important to realize that there are several Transformer variants.  The original code places `LayerNorm` layers
after each transform, and uses sinusoidal (fixed) positional embeddings.  

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
  - *ConveRT relative attention* - TODO

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

We provide an addon on mead-hub to use the excellent HuggingFace libraries to fine-tune models like BERT in MEAD.  This
addon simply overrides

#### Fine-Tuning BERT Official

We provide an addon on mead-hub to use either the BERT official code (a copy) or the TF Hub module to fine-tune.
Since this is based on the official code, no conversion or non-standard checkpoints are required.



## Pre-training your own Transformers with MEAD

### Using the API Examples

MEAD has a rich set of pre-training solutions provided in the API examples, with support for a wide-variety of
Transformer-based pre-training.  For pre-training, which often requires training multi-GPU, multi-worker, we found
it easiest to use the PyTorch DistributedDataParallel facilities to train, and to have the programs export a
framework-independent NPZ file that can be loaded by either TensorFlow or PyTorch MEAD layers.

#### Pre-training Transformer-based Language Models with MEAD

By now, the practice of pre-training Transformer models on a large amount of data to be used for downstream fine-tuning
is ubiquitous.  MEAD supports pre-training architectures such as BERT/Roberta via several scripts, all of which
support multi-worker training.

* [pretrain_tlm_pytorch](../api-examples/pretrain_tlm_pytorch.py)*: Train with multi-worker configuraion with a large
(possibly out-of-core) LM dataset, using either `fastBPE` or `WordPiece` to tokenize the words (PyTorch).
This program supports export of torch checkpoints as well as NPZ checkpoints, and supports reloading from either.
Supports Kubernetes-based training with PyTorchJob operator

* [pretrain_tlm_tf](../api-examples/pretrain_tlm_tf.py)*: Train multi-worker on TPUs or GPUs with a large
(possibly out-of-core) LM dataset, using [fastBPE](https://github.com/glample/fastBPE) to tokenize the words.  This program supports export of TF checkpoints
as well as NPZ checkpoints.  Supports training on TPUs as well as Kubernetes-based training with either TFJob operator, Job or Pod CRDs.
  - There is a [stripped down example Google Colab tutorial](https://colab.research.google.com/github/dpressel/mead-tutorials/blob/master/mead_transformers_tpu.ipynb) based on this program where you can train an MLM on a TPU

* [pretrain_paired](../api-examples/pretrain_paired.py)*: Train either a Transformer encoder-decoder or a Transformer
dual encoder out-of-core on very large datasets.  The dual-encoder follows [ConveRT, Henderson et al 2019](https://arxiv.org/pdf/1911.03688.pdf).
This variant creates 2 context, one for the encoder and one for the
decoder, either based on splitting a single line in 2 (Next Sequence Prediction or NSP), or by separating the line 
by a tab delimiter (Next Turn Prediction or NTP).  In the latter case, its presumed that the separator will partition
the line into a query and a response.

* [pretrain_discrim](../api-examples/pretrain_discrim.py)*: In this approach, we jointly train a generator and a
discriminator following [ELECTRA: Pre-Training Text Encoders as Discriminators Rather Than Generators,
    Clark et al. 2019](https://openreview.net/pdf?id=r1xMH1BtvB).

#### Writing your own Pre-training script

Its pretty easy to tweak the above examples to create your own new training regime with Transformers.  Copy one
of the other training examples that is closest to what you are trying to do, and modify the `create_loss()` and or
the model architecture accordingly, and kick it off on your machine (or a kubernetes cluster) and make sure you
follow the structure of the other examples.  If you can use the existing baseline models 
`TransformerMaskedLanguageModel` or `TransformerLanguageModel`, this will make things the easiest.  If not, you can
create a similar class to one of those and change the portions you need (or subclass them).

- If you need to create your own Transformer LM
  - make sure that the internal structure of the encoder is named either `generator` or `transformer` so that the 
  serializers/converters know where to find them.
  - make sure to use an `EmbeddingsStack` named `embeddings` to contain your embeddings objects, and use some form of
  `*PositionalEmbeddingsModel` or `LookupTableEmbeddingsModel` for your embeddings where possible to avoid having to
  create custom serializers/converters

### Transformer Checkpoint conversion

