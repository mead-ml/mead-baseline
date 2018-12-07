# Sentence Classification

There are several models built in to the `baseline` codebase.  These are summarized individually in the sections below

## Convolution Model

This code provides PyTorch, TensorFlow, DyNet and Keras implementations. 

*Details*

This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and before that Collobert's "Sentence Level Approach."  The implementations provided here are basically the Kim static and non-static models.

Temporal convolutional output total number of feature maps is configurable (this also defines the size of the max over time layer, by definition). This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd, with and without momentum).  The Kim paper uses adadelta, which works well, but vanilla SGD and Adam often work well.

Despite the simplicity of this approach, on many datasets this model performs better than other strong baselines such as NBSVM.
There are some options in each implementation that might vary slightly, but this approach should do at least as well as the original paper.

### With Fine-tuning Embedding (LookupTable) layer

The (default) fine-tuning approach loads the word2vec weight matrix into an Lookup Table.  As we can see from the Kim paper, dynamic/fine-tuning embedding models do not always out-perform static models, However, they tend to do better.

We randomly initialize unattested words and add them to the weight matrix for the Lookup Table.  This can be controlled with the 'unif' parameter in the driver program.

### Static, "frozen" Embedding (LookupTable) layer

The static (no fine-tuning) model usually has decent performance, and the code is very simple to implement from scratch, as long as you have access to a fast convolution operator.  For most cases, fine-tuning is preferred.

### Running It

Early stopping with patience is supported.  There are many hyper-parameters that you can tune, which may yield many different models.

[Here](../python/mead/config/sst2.json) is a tensorflow example running Stanford Sentiment Treebank 2 data with adadelta:

```
python trainer.py --config config/sst2.json
```

### Latest Runs

Here are the last observed performance scores using with fine-tuning on the Stanford Sentiment Treebank 2 (SST2), dbpedia and TREC-QA (config [here](../python/mead/config/trec-cnn.yml)):

| Dataset | Results    |
| ------- | ---------- | 
| sst2    |       87.9 |
| dbpedia |     99.054 | 
| trec-qa |       94.6 |
| AG      |       92.5 |

Note that these are randomly initialized and these numbers will vary
(IOW, don't assume that one implementation is guaranteed to outperform the others from a single run).

On a typical GPU, each implementation for SST2 takes between 15 - 30s per epoch depending on the deep learning framework (TensorFlow and PyTorch are fastest, and about the same speed)

## LSTM Model

Provides a simple LSTM and BLSTM for text classification with PyTorch, TensorFlow and DyNet

*Details*

The LSTM's final hidden state is passed to the final layer.  The use of an LSTM instead of parallel convolutional filters is the main differentiator between this model and the default model (CMOT) above.  To request the LSTM classifier instead of the default, pass `"model_type": "lstm"` to the driver program.

### Running It

This model is run similarly to the model above.
[Here](../pythom/mead/config/sst2-lstm.json) is an example running Stanford Sentiment Treebank 2 data with adadelta using TensorFlow:

```
python trainer.py --config config/sst2-lstm.json
```

## Neural Bag of Words (NBoW) Model (Max and Average Pooling)

Two different pooling methods for NBoW are supported: max (`"model_type": "nbowmax"`) and average (`"model_type": "nbow"`).  Passing `"layers": <N>` defines the number of hidden layers, and passing `"hsz": <HU>` defines the number of hidden units for each layer.

### Status


This model is implemented in TensorFlow, PyTorch and DyNet. Multi-GPU support can be enabled by passing `--gpus <N>` to the `mead-train`.  `--gpus -1` is a special case that tells the driver to run with all available GPUs.  The `CUDA_VISIBLE_DEVICES` environment variable should be set to create a mask of GPUs that are visible to the program.

### Latest Runs

Here are the last observed performance scores using _classify_sentence_ with fine-tuning on the Stanford Sentiment Treebank 2 (sst2) with a single hidden layer (the default), and `hsz=100` (the default)

It was run on the latest code as of 8/24/2017, with 25 epochs with adadelta as an optimizer:

#### NBoW Results

| model_type | Dataset | TensorFlow | PyTorch | 
|------------| ------- | ---------- | ------- | 
| nbowmax    | sst2    |       82.8 |  84.1   |
| nbow       | sst2    |       84.2 |  82.9   |

Note that these are randomly initialized and these numbers will vary
(IOW, don't assume that one implementation is guaranteed to outperform the others from a single run).


#### Losses and Reporting

When training the loss that is optimized is the total loss averaged over the number of examples in the mini-batch.

When reporting the loss reported every nsteps is the total loss averaged over the number of examples that appeared in these nsteps number of minibatches.

When reporting the loss at the end of an epoch it is the total loss averaged over the number of examples seen in the whole epoch.

Metrics like accuracy and f1 are computed at the example level.
