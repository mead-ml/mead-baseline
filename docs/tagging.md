# Sequence Tagging

There are several models built in to the `baseline` including BiLSTM, Transformer and ConvNet approaches.

The documentation here includes information on these models and the expected results as well as some documentation on the API design and how to make your own models


## Sequence tagging using CNN-BLSTM-CRF

Our Baseline Tagger is a State-of-the-Art model architecture, and supports flexible embeddings, including contextual embeddings and one or more pre-trained sets of word embeddings.  It has been shown that character-level modeling is important in deep models to support morpho-syntatic structure for tagging tasks.

### CNN-BLSTM-CRF

The code uses word and character-level word embeddings.  For character-level processing, a character vector depth is selected, along with a word-vector depth. 

The character-level embeddings are based on Character-Aware Neural Language Models (Kim, Jernite, Sontag, Rush) and dos Santos 2014 (though the latter's tagging model is quite different).  Unlike dos Santos' approach, here, one or more parallel filters are applied during the convolution (which is like the Kim approach). Unlike the Kim approach residual connections of like size filters are used, and since they improve performance for tagging, word vectors are also used.

## Part-of-Speech Tagging

Twitter is a challenging data source for tagging problems.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, and a very coarse approach to compositional character to word modeling significantly out-performs this.

To run our default model:

```
python --config config/twpos.json
```

To change the backend between TensorFlow and PyTorch, just change the `backend` parameter (to `tensorflow` or `pytorch`).  For DyNet, we employ autobatching, which also affects the reader, so in addition to setting the `backend` to `dynet`, set the `batchsz` to `1`, and then set the `autobatchsz` under the `train` parameters to the desired batch size. See the example configs under [mead](../python/mead/config) for details.

## NER Tagging


For NER reporting, we report an F1 score on at each validation pass, and use F1 for early-stopping.

For tasks that require global coherency like NER tagging, it has been shown that using a linear chain CRF to learn a transition matrix between label states in conjunction with the output RNN tags improves performance.  A simple technique that performs nearly as well is to replace the CRF with a coarse constrained decoder at inference time that simply checks for IOBES violations and effectively zeros out those transitions.

### CONLL2003

#### CRF Model

Our default [mead](mead.md) configuration is SoTA for models without contextual embeddings or co-training, and is run with this command:

```
mead-train --config config/conll.json
```

- We find (along with most other researchers) that IOBES (BMES), on average, out-performs BIO (IOB2).  We also find that IOB2 out-performs IOB1 (the original format in which the data is provided).  Note these details do not change the actual model itself, it just trains the model in a way that seems to cause it to learn better, and when comparing implementations, its important to take note of which of the three formats are used.

#### Constrained Decoding Model without CRF

Our constrained decoder can be run with the following command:

```
mead-train --config config/conll-no-crf.json
```

### WNUT17

WNUT17 is a task that deals with NER for rare and emerging entities on Twitter.  Scores are typically much lower on this task than for traditional NER tagging

#### CRF

Our default model provides a strong baseline for WNUT17, and  can be trained with this command:

```
mead-train --config config/wnut.json
```

#### Constrained Decoding Model without CRF

Our constrained decoder model can be run as follows:

```
mead-train --config config/wnut-no-crf.json
```

### Ontonotes 5.0

Ontonotes 5.0 is a NER dataset that is larger than CONLL2003 and contains more entity types. Dataset was created with Ontonotes 5.0 using the data splits from the [CONLL2012 shared task](http://conll.cemantix.org/2012/data.html) version 4 for train/dev/test.

Out default CRF model can be run with the following:

```
mead-train --config config/ontonotes.json
```

### SNIPS NLU slot filling

SNIPS NLU is data created by SNIPS for benchmarking chatbots systems. It includes intent detection and slot filling, This is the slot filling model and can be trained with the following:

```
mead-train --config config/snips.json
```

### Model Performance

We have done extensive testing on our tagger architecture.  Here are the results after runing each model 10 times for each configuration.  All NER models are trained with IOBES tags.  The ELMo configurations are done using our [ELMo Embeddings addon](../python/addons/embed_elmo.py).

| Config                                                       | Dataset   | Model                | Metric | Mean  |  Std  | Min   | Max   |
|:-------------------------------------------------------------|:----------|:---------------------|-------:|------:|------:|------:|------:|
| [twpos.json](../python/mead/config/twpos.json)               | twpos-v03 | CNN-BLSTM-CRF        |    acc | 90.75 | 0.140 | 90.53 | 91.02 |
| [conll.json](../python/mead/config/conll.json)               | CONLL2003 | CNN-BLSTM-CRF        |     f1 | 91.47 | 0.247 | 91.15 | 92.00 |
| [conll-no-crf.json](../python/mead/config/conll-no-crf.json) | CONLL2003 | CNN-BLSTM-constrain  |     f1 | 91.44 | 0.232 | 91.17 | 91.90 |
| [conll-elmo.json](../python/mead/config/conll-elmo.json)     | CONLL2003 | CNN-BLSTM-CRF        |     f1 | 92.26 | 0.157 | 92.00 | 92.48 |
| [wnut.json](../python/mead/config/wnut.json)                 | WNUT17    | CNN-BLSTM-CRF        |     f1 | 40.33 | 1.13  | 38.38 | 41.90 |
| [wnut-no-crf.json](../python/mead/config/wnut-no-crf.json)   | WNUT17    | CNN-BLSTM-constrain  |     f1 | 40.59 | 1.06  | 37.96 | 41.71 |
| [ontonotes.json](../python/mead/config/ontonotes.json)       | ONTONOTES | CNN-BLSTM-CRF        |     f1 | 87.41 | 0.166 | 87.14 | 87.74 |
| [snips.json](../python/mead/config/snips.json)               | SNIPS     | CNN-BLSTM-CRF        |     f1 | 95.55 | 0.394 | 94.85 | 96.07 |

### Testing a trained model on your data

You can use [`tag-text.py`](../api-examples/tag-text.py) to load a sequence tagger checkpoint and predict its labels

#### Losses and Reporting

Loss functions are defined by the decoders.  If a CRF is used then the loss is the CRF loss averaged over the number of examples in the mini-batch. When using a word level loss the loss is the sum of the cross entropy loss of each token averaged over the number of examples in the mini-batch. Both of these are batch level losses.

When reporting the loss every nsteps for the CRF the loss is the total CRF loss divided by the number of examples in the last nstep number of mini-batches. For word level loss it is the total word level loss divided by the number of examples in the last nstep number of batches.

The epoch loss is the total loss averaged over the total number of examples in the epoch.

Accuracy is computed on the token level and F1 is computed on the span level.


## API Design & Architecture

There is a base interface `TaggerModel` defined for taggers inside of baseline/model.py and a specific framework-dependent sub-class `TaggerModelBase` that implements that interface and also implements the base layer class defined by the underlying deep-learning framework (a `Model` in Keras and `Module` in PyTorch).

The `TaggerModelBase` provides fulfills the `create()` function required in the base interface (`TaggerModel`) but leaves an abstract method for creation of the layers: `create_layers()` and leaves an abstract forward method (`forward()` for PyTorch and `call()` for Keras).

While this interface provides lots of flexibility, its sub-class `TransducerTaggerModel(TaggerModelBase)` provides much more utility and structure to tagging and as typically the best place to extend from when writing your own tagger.

The `TaggerModelBase` follows a single basic pattern for modeling tagging consisting for 3 basic steps:

1. embeddings
2. encoder (transduction and projection to final number of labels)
3. decoder (typically a constrained greedy decoder or a CRF)

All of the MEAD-Baseline tagger models reuse steps 1. and 3. and define their own encoders by overriding the `init_encode()` method.  These hooks are called from the concrete implementation of `create_layers()`, and the forward method is implemented by a simple flow that executes these layers.

Most taggers can be composed together by providing the encoder layer and wiring in a set of Embeddings (usually consisting of a combo of word features and word character-compositional features concatenated together), a decoder (typically a `GreedyTaggerDecoder` or a `CRF`).  For example, a CNN-BiLSTM-CRF can be composed by providing:

1. embeddings via concatenation of `LookupTableEmbeddingsModel` and `CharConvEmbeddingsModel`
2. encoder via `RNNTaggerModel`
3. decoder via `CRF`

For fine-tuning a language model like BERT, the typical approach is to remove the head from the model (AKA the final linear projection out to the vocab and the normalizing softmax), and to put a final linear projection to the output number of classes in its place.  In MEAD-Baseline, the headless LM would be lodaded as an embedding object and passed into an `EmbeddingsStack` so the encoder itself should be pass through.


### Writing your own tagger


#### Some best practices for writing your own tagger

- Do not override the `create()` from the base class unless absolutely necessary.  `create_layers()` is the typical extension point for the `TaggerModelBase`
- Use the mead layers (8 mile) API to define your layers
  - This will minimize any incompatibility and make it easy to switch frameworks later
- Use `TransducerTaggerModel` for simple models involving overriding the `init_encode()` method.  If you can do this, avoid overidding `create_layers()`

### Some Notes on the TensorFlow implementation

### Eager vs Declarative

In TensorFlow, we are supporting eager and declarative mode models.  This fundamentally changes how our objects must look, as in declarative mode, we must do a `session.run` identifying the graph outputs, vs eager mode, which can look nearly identical to PyTorch.

If you wanted to write code to only support graph mode, especially prior to Keras integration, it was common to simply define the graph procedurally and expose its graph endpoints to the class for future use in `session.run`.  This is what previous versions of MEAD-Baseline did.  With Keras now becoming the primary and recommended interface to building neural networks in TensorFlow execution occurs in essentially 2 phases (initialization and execution) corresponding to creation of the layer by calling its constructor vs execution of the `call()` operator `layer(x)`.  In declarative mode, however, ultimately we do need something to call `session.run` on, so we typically will save the output as a property on the class that can be handed to that call.  In eager mode, storing an output property graph node makes no sense at all, so we just dont do it.  Also in eager mode with 2.x, we save and load models somewhat differently than in TF 1.x.

Eager mode massively simplifies the experience of TensorFlow users, however it is not completely trivial to support a training loop for both side-by-side.  In an ideal world where everyone uses eager, we could just use a normal for loop over the input dataset to train the model with gradient tape.  Of course, this doesnt work with declarative mode.  We decided that the cleanest way to support both methods was to have a single classifier model, but to have multiple custom training loops that are defined as `fit_func`s.  The default `fit_func` for eager mode is the previously mentioned simple for loop with gradient tape whereas, for declarative model it a dataset based method relying on iterators.  In V1.x of MEAD-Baseline, the default `fit_func` was a `feed_dict` method.  We still support this method of training by providing a built-in `fit_func` with key `feed_dict`.

When `mead-train` runs in declarative mode (which is controlled by the `--prefer_eager <bool>` argument), the training block of the MEAD config can contain a key `fit_func: feed_dict` to switch the training loop implementation to the previous default method (this can also be done at run-time by passing `--fit_func feed_dict` as a command-line argument to `mead-train`.
