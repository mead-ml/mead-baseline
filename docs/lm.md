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
