# Language Modeling with Recurrent Neural Networks

The Language Modeling code supports several common baselines:

- Transformer LMs
  - MLMs
  - Causal LMs

- RNNLMs
  - Recurrent Neural Network Regularization (Zaremba, Vinyals, Sutskever) (2014)
    - https://arxiv.org/pdf/1409.2329.pdf
- RNNLMs with character-compositional features
  - Character-Aware Neural Language Models (Kim, Jernite, Sontag, Rush)
     - https://arxiv.org/pdf/1508.06615.pdf
  - Embedding dropout and variational RNNs
  
To run the Zaremba model with their "medium regularized LSTM" configuration, early stopping, and pre-trained word vectors:


```
mead-train --config config/ptb-med.json
```

## Status

TODO: out-of-date! re-benchmark and update this!

The "medium regularized LSTM" above (_Word Med_ below) has a lower perplexity than the original paper (even the large model).
As noted above, the run above differs in that it uses pre-trained word vectors.

|        Model       | Framework  | Dev    |  Test   |
| ------------------ | ---------- | ------ | ------- |
| Word Med (Zaremba) | TensorFlow | 80.168 | 77.2213 |

_TODO: Add LSTM Char Small Configuration results_

#### Losses and Reporting

The loss that is optimized is the total loss divided by the total number of tokens in the mini-batch (token level loss). This is different than how the loss is calculated in Tensorflow Tutorial but it is how the loss is calculated in awd-lm ([Merity et. al, 2017](https://arxiv.org/abs/1708.02182)), Elmo ([Peters et. al., 2018](https://arxiv.org/abs/1802.05365)), OpenAI GPT ([Radford et. al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)), and BERT ([Devlin et. al., 2018](https://arxiv.org/pdf/1810.04805.pdf))

When reporting the loss every nsteps it is the total loss divided by the total number of tokens in the last nstep number of mini-batches. The perplexity is e to this loss.

The epoch loss is the total loss averaged over the total number of tokens in the whole epoch. The perplexity is e to this loss. This results in token level perplexity which is standard reporting in the literature.



## API Design & Architecture

There is a base interface `LanguageModel` defined for LMs inside of baseline/model.py and a specific framework-dependent sub-class `LanguageModelBase` that implements that interface and also implements the base layer class defined by the underlying deep-learning framework (a `Model` in Keras and `Module` in PyTorch).

The `LanguageModelBase` provides fulfills the `create()` function required in the base interface (`LanguageModel`) but leaves an abstract method for creation of the layers: `create_layers()` and leaves an abstract forward method (`forward()` for PyTorch and `call()` for Keras).

While this interface provides lots of flexibility, its sub-class `AbstractGeneratorLanguageModel(LanguageModelBase)` provides much more utility and structure to tagging and as typically the best place to extend from when writing your own tagger.

The `AbstractGeneratorLanguageModel` follows a single basic pattern for modeling:

1. embeddings
2. embeddings_proj (project the embeddings to a different dimension)
  - This allows us to do things like embeddings decomposition where the LUT output dims are much lower than the `d_model/hsz` for the generator, which is computationally efficient
3. generator a hidden representation given the previous tokens
4. projection to logits space (the number of labels)
  - This allows weight tying on the output as long as the input to hidden projection is not composed of a decomposition (IOW, the embeddings `output_dim` or `dsz` must be the same as the LM hidden size (`hsz`)

Most LM can be composed together by providing the geneator layer and wiring in a set of Embeddings (usually consisting of a combo of word features and word character-compositional features concatenated together) and providing information about the embedding feature to be used for output token generation

### Writing your own language model

#### Some best practices for writing your own language model

- Do not override the `create()` from the base class unless absolutely necessary.  `create_layers()` is the typical extension point for the `LanguageModelBase`
- Use the mead layers (8 mile) API to define your layers
  - This will minimize any incompatibility and make it easy to switch frameworks later
- Use `AbstractGeneratorLanguageModel` for simple models involving overriding the `init_encode()` method.  If you can do this, avoid overidding `create_layers()`
- Make sure to derive the `requires_state` property if you are inheriting from the abstract generator, so that the trainer will know whether or not to initialize the state on creation

### Some Notes on the TensorFlow implementation

### Eager vs Declarative

In TensorFlow, we are supporting eager and declarative mode models.  This fundamentally changes how our objects must look, as in declarative mode, we must do a `session.run` identifying the graph outputs, as well as defining a `zero_state` tensor that can be used to re-initialize truncate backprop.  In eager mode, however, the models look nearly identical, as does the training loop.

If you wanted to write code to only support graph mode, especially prior to Keras integration, it was common to simply define the graph procedurally and expose its graph endpoints to the class for future use in `session.run`.  This is what previous versions of MEAD-Baseline did.  With Keras now becoming the primary and recommended interface to building neural networks in TensorFlow execution occurs in essentially 2 phases (initialization and execution) corresponding to creation of the layer by calling its constructor vs execution of the `call()` operator `layer(x)`.  In declarative mode, however, ultimately we do need something to call `session.run` on, so we typically will save the output as a property on the class that can be handed to that call.  In eager mode, storing an output property graph node makes no sense at all, so we just dont do it.  Also in eager mode with 2.x, we save and load models somewhat differently than in TF 1.x.

Eager mode massively simplifies the experience of TensorFlow users, however it is not completely trivial to support a training loop for both side-by-side.  In an ideal world where everyone uses eager, we could just use a normal for loop over the input dataset to train the model with gradient tape.  Of course, this doesnt work with declarative mode.  We decided that the cleanest way to support both methods was to have a single classifier model, but to have multiple custom training loops that are defined as `fit_func`s.  The default `fit_func` for eager mode is the previously mentioned simple for loop with gradient tape whereas, for declarative model it a dataset based method relying on iterators.  In V1.x of MEAD-Baseline, the default `fit_func` was a `feed_dict` method.  We still support this method of training by providing a built-in `fit_func` with key `feed_dict`.

When `mead-train` runs in declarative mode (which is controlled by the `--prefer_eager <bool>` argument), the training block of the MEAD config can contain a key `fit_func: feed_dict` to switch the training loop implementation to the previous default method (this can also be done at run-time by passing `--fit_func feed_dict` as a command-line argument to `mead-train`.
