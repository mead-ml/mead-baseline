# Sentence Classification

There are several models built in to the `baseline` codebase.  These are summarized individually in the sections below, and an overall performance summary is given at the bottom

## A Note About Fine-Tuning and Embeddings

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
python trainer.py --config config/sst2.json
```

## LSTM Model

*Details*

The LSTM's final hidden state is passed to the final layer.  The use of an LSTM instead of parallel convolutional filters is the main differentiator between this model and the default model (CMOT) above.  To request the LSTM classifier instead of the default, set `"model_type": "lstm"` in the mead config file.

The command below executes an LSTM classifier with 2 sets of pre-trained word embeddings

```
python trainer.py --config config/sst2-lstm.json
```

## Neural Bag of Words (NBoW) Model (Max and Average Pooling)

Two different pooling methods for NBoW are supported: max (`"model_type": "nbowmax"`) and average (`"model_type": "nbow"`).  Passing `"layers": <N>` defines the number of hidden layers, and passing `"hsz": <HU>` defines the number of hidden units for each layer.

## Classifier Performance

We run each experiment 10 times and list the performance, configuration, and metrics below

| config                                                           | dataset   | model                | metric | mean  |  std  | min   | max   |
| ---------------------------------------------------------------- | --------- | -------------------- |------- | ------| ----- | ----- | ----- |
| [sst2-lstm.json](../python/mead/config/sst2-lstm.json)           | SST2      | LSTM 2 Embeddings    |    acc | 88.57 | 0.443 | 87.59 | 89.24 |
| [sst2-lstm-840b.json](../python/mead/config/sst2-lstm-840b.json) | SST2      | LSTM 1 Embedding     |    acc | 88.39 | 0.45  | 87.42 | 89.07 |
| [sst2.json](../python/mead/config/sst2.json)                     | SST2      | CNN-3,4,5            |    acc | 87.32 | 0.31  | 86.60 | 87.58 |
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
