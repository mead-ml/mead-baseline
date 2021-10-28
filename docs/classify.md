# Text Classification

There are several classification models built in to the codebase.  These are summarized individually in the sections below, and an overall performance summary is given at the bottom.
The final section provides information on the API design, and how to make your own models

## A Note About Fine-Tuning Embeddings

For the lookup-table embeddings, you can control whether or not the embeddings should be fine-tuned by passing a boolean `finetune` in for the `embeddings` section of the mead config.  If you are using random weights, you definitely should fine tune.  If you are using pre-trained embeddings, it may be worth experimenting with this option.  The default behavior is to fine-tune embeddings.  We randomly initialize unattested words and add them to the weight matrix for the Lookup Table.  This can be controlled with the 'unif' parameter in the driver program.

## Convolution Model

*Details*

This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and before that Collobert's "Sentence Level Approach."  The implementations provided here are basically the Kim static and non-static models.

Temporal convolutional output total number of feature maps is configurable (this also defines the size of the max over time layer, by definition). This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd, with and without momentum).  The Kim paper uses adadelta, which works well, but vanilla SGD and Adam often work well.

Despite the simplicity of this approach, on many datasets this model performs better than other strong baselines such as NBSVM.
There are some options in each implementation that might vary slightly, but this approach should do at least as well as the original paper.

Early stopping with patience is supported.  There are many hyper-parameters that you can tune, which may yield many different models.

To run, use this command:

```
mead-train --config config/sst2.json
```

## LSTM Model

*Details*

The LSTM's final hidden state is passed to the final layer.  The use of an LSTM instead of parallel convolutional filters is the main differentiator between this model and the default model (CMOT) above.  To request the LSTM classifier instead of the default, set `"model_type": "lstm"` in the mead config file.

The command below executes an LSTM classifier with 2 sets of pre-trained word embeddings

```
mead-train --config config/sst2-lstm.json
```

## Neural Bag of Words (NBoW) Model (Max and Average Pooling)

Two different pooling methods for NBoW are supported: max (`"model_type": "nbowmax"`) and average (`"model_type": "nbow"`).  Passing `"layers": <N>` defines the number of hidden layers, and passing `"hsz": <HU>` defines the number of hidden units for each layer.

## Fine-Tuning Models

These models are just defined as a final layer on top of some pre-trained, pooled embedding representation, but may also include additional MLP layers.
[This document has a more in-depth discussion](fine-tuning.md) of fine-tuning in MEAD/Baseline.

For example, we can fine-tune BERT using

```
mead-train --config config/sst2-bert-base-uncased.yml
```


## Classifier Performance

We run each experiment 10 times and list the performance, configuration, and metrics below

| config                                                           | dataset   | model                | metric | mean  |  std  | min   | max   |
| ---------------------------------------------------------------- | --------- | -------------------- |------- | ------| ----- | ----- | ----- |
| [sst2-lstm.json](../python/mead/config/sst2-lstm.json)           | SST2      | LSTM 2 Embeddings    |    acc | 88.57 | 0.443 | 87.59 | 89.24 |
| [sst2-lstm-840b.json](../python/mead/config/sst2-lstm-840b.json) | SST2      | LSTM 1 Embedding     |    acc | 88.39 | 0.45  | 87.42 | 89.07 |
| [sst2.json](../python/mead/config/sst2.json)                     | SST2      | CNN-3,4,5            |    acc | 87.32 | 0.31  | 86.60 | 87.58 |
| [sst2-bert-base-uncased.yml](../python/mead/config/sst2-bert-base-uncased.yml)                     | SST2      | BERT-Base Fine-tuned |    acc | 93.45 | -     | -     | 94.07 |
| [trec-cnn.yml](../python/mead/config/trec-cnn.yml)               | TREC-QA   | CNN-3                |    acc | 92.33 | 0.56  | 91.2  | 93.2  |
| [ag-news-lstm.json](../python/mead/config/ag-news-lstm.json)     | AGNEWS    | LSTM 2 Embeddings    |    acc | 92.60 | 0.20  | 92.3  | 92.86 |
| [ag-news.json](../python/mead/config/ag-news.json)               | AGNEWS    | CNN-3,4,5            |    acc | 92.51 | 0.199 | 92.07 | 92.83 |

## Multiple GPUs

Multi-GPU support can be setting the `CUDA_VISIBLE_DEVICES` environment variable to create a mask of GPUs that are visible to the program.


## Losses and Reporting

When training the loss that is optimized is the total loss averaged over the number of examples in the mini-batch.

When reporting the loss reported every nsteps is the total loss averaged over the number of examples that appeared in these nsteps number of minibatches.

When reporting the loss at the end of an epoch it is the total loss averaged over the number of examples seen in the whole epoch.

Metrics like accuracy and f1 are computed at the example level.

## API Design & Architecture

There is a base interface `ClassifierModel` defined for classifiers inside of baseline/model.py

This defines the base API for usage, but doesnt provide any implementation.  Each framework sub-module defines a sub-class of this callled `ClassifierModelBase` that all sub-models are built from.  Both Keras and Torch have a base model concept that needs to be extended in this base class.  In Keras, this is the `tf.keras.Model` and in PyTorch, it is the `nn.Module`.  Ultimately, for both Keras and PyTorch, we will need to define a function to create the layers (`create_layers`) and to perform the forward step for these layers (In PyTorch, this is `forward` and in Keras this is `call`.

Most deep learning NLP classifiers tend to reduce to a few simple idioms.  For fine-tuning a language model like BERT, for example, the typical approach is to remove the head from the model (AKA the final linear projection out to the vocab and the normalizing softmax), and to put a final linear projection to the output number of classes in its place.  In MEAD-Baseline, the headless LM would be lodaded as an embedding object and passed into an `EmbeddingsStack` so the model finally just needs an output layer (and optionally, some MLP intermediate layers).

The `FineTuneModelClassifier(ClassifierModelBase)` model provides this interface by overloading `create_layers()` to produce the penultimate stacking layers (if needed) and the output layer.

For cases where we build an actual model from scratch on top of e.g. word embeddings, MEAD-Baseline the typical pattern can be reduced to:

1. embedding
2. pooling to a fixed representation
3. optional MLP stacking
4. output layer

We provide a sub-class `EmbedPoolStackClassifier(ClassifierModelBase)` that fulfills this interface by extending `create_layers()` and providing `init_*` functions to create sub-layers for each of these steps, and providing a reasonable implementation of all but the pooling layer.

We extend this model with our previously described Baseline models by simply overriding the pooling layer.

By effectively separating out the concerns of layer creation, the formation of membeddings, saving and loading of features, our derived sub-classes are nearly identical between PyTorch and TensorFlow.

### Writing your own classifier

The only real requirement to provide mead with your own classifier is to extend the ClassifierModelBase from the framework you are using, and to register your model with the `@register_model` annotation to identify a key that uniquely identifies your model.  However, there are some things that make it very easy or succinct to define custom models that we recommend.

#### Some best practices for writing your own classifier

- Use the mead layers (8 mile) API to define your layers
  - This will minimize any incompatibility and make it easy to switch frameworks later
  - Determine if your model can sub-class the `EmbedPoolStackClassifer`.  If it follows this idiom, it may be very succinct to define a new classifer by overriding a single function (typically the `init_pool()` function).
- If you are overriding the `EmbedPoolStackClassifier`, remember that the pooling layer is expecting 2 arguments -- the input tensor itself and a tensor containing the length.  If you want to adapt an existing Layer from Keras/PyTorch or elsewhere that requires a single input Tensor, use the `WithoutLengths(YourLayer)` adapter which strips the length tensor and provides a single input tensor to `YourLayer`

### Some Notes on the TensorFlow implementation

### Eager vs Declarative

In TensorFlow, we only supporting eager mode models (previous versions also support declarative mode).

If you wanted to write code to only support graph mode, especially prior to Keras integration, it was common to simply define the graph procedurally and expose its graph endpoints to the class for future use in `session.run`.  This is what previous versions of MEAD-Baseline did.  With Keras now becoming the primary and recommended interface to building neural networks in TensorFlow execution occurs in essentially 2 phases (initialization and execution) corresponding to creation of the layer by calling its constructor vs execution of the `call()` operator `layer(x)`.
