# Structured Prediction using CNN-BLSTM-CRF

This code is useful for tagging tasks, e.g., POS tagging, chunking and NER tagging.  Recently, several researchers have proposed using RNNs for tagging, particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A common modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.  This usually improves the resolution of the tagger significantly.  That is the default approach taken here.

To execute these models it is necessary to form word vectors.  It has been shown that character-level modeling is important in deep models to support morpho-syntatic structure for tagging tasks.
It seems that a good baseline should combine word vectors and character-level compositions of words.

## CNN-BLSTM-CRF

The code uses word and character-level word embeddings.  For character-level processing, a character vector depth is selected, along with a word-vector depth. 

The character-level embeddings are based on Character-Aware Neural Language Models (Kim, Jernite, Sontag, Rush) and dos Santos 2014 (though the latter's tagging model is quite different).  Unlike dos Santos' approach, here parallel filters are applied during the convolution (which is like the Kim approach). Unlike the Kim approach residual connections of like size filters are used, and since they improve performance for tagging, word vectors are also used.

Twitter is a challenging data source for tagging problems.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, and a very coarse approach to compositional character to word modeling still gets equivalent results.

## Running It

Here is an example using convolutional filters for character embeddings, alongside word embeddings.  This is basically a combination of the dos Santos approach with the Kim parallel filter idea:

```
python --config config/conll-twpos.json
```
To run with PyTorch, just pass `--backend pytorch`

### NER (and other IOB-type) Tagging

NER tagging can be performed with a BLSTM, and optionally, a top-level CRF. This will report an F1 score on at each validation pass, and will use F1 for early-stopping as well.

For tasks that require global coherency like NER tagging, it has been shown that using a transition matrix between label states in conjunction with the output RNN tags improves performance.  This makes the tagger a linear chain CRF, and we can do this by simply adding another layer on top of our RNN output.

The [mead](mead.md)  configuration below yields a very similar model to this paper: https://arxiv.org/pdf/1603.01354.pdf and yields near-SoTA performance.

```
python trainer.py --config config/conll-iobes.json
python trainer.py --config config/conll-bio.json
```

- We find (along with most other papers/implementors) that IOBES (BMES), on average, out-performs BIO (IOB2).  We also find that IOB2 out-performs IOB1 (the original format in which the data is provided).  Note these details do not change the actual model itself, it just trains the model in a way that seems to cause it to learn better, and when comparing implementations, its important to take note of which of the three formats are used.

## Status

This model is implemented in TensorFlow and PyTorch.

### Latest Runs

Here are some observed performance scores on various dataset.  In general, our observed peformance is very similar to this recent paper: https://arxiv.org/pdf/1806.04470.pdf


| dataset             | metric | method   | eta (LR) |    avg | proj | crf  | hsz |
| ------------------- | ------ | -------- | -------  | ------ | -----| -----|-----|
| twpos-v03           |    acc | adam     |       -- | 89.4   | N    | N    | 100 |
| conll 2003 (IOB1)   |     f1 | sgd mom. |     0.015| 90.8   | N    | Y    | 200 |
| conll 2003 (BIO)    |     f1 | sgd mom. |     0.015| 90.9   | N    | Y    | 200 |
| conll 2003 (IOB2)   |     f1 | sgd mom. |     0.015| 91.1   | N    | Y    | 200 |
|       atis (mesnil) |     f1 | sgd mom. |     0.01 | 96.74  | N    | N    | 100 |

### Testing a trained model on your data

You can use [`tag.py`](../python/tag.py) to load a sequence tagger checkpoint, predict the labels for an input conll file and produce the output in the same format. The second column is the predicted label.

Run the code like this:
```
python tag.py --input conll-tester.conll --output out-conll-tester.conll --model mead/tagger/tagger-model-tf-7097 --mxlen 124 --mxwlen 61
```
`tag.py` also supports multiple features in the conll file. However, you need to create a JSON file (dictionary) with the index of the features in the format: `{feature-name:feature-index}` (eg. `{"gaz":1}`) where the `gaz` feature is the second column in the conll file. To run the code use: 

```
python tag.py --input gaz-tester.conll --output gaz-tester-out-1.conll --model mead/tagger-gaz/tagger-model-tf-12509 --mxlen 60 --mxwlen 40 --model_type gazetteer --features features.json --featurizer_type multifeature
```

You can write your own featurizer (`featurizer_<featurizer_type>.py`) and keep it in your PYTHONPATH. For an example, see [featurizer_elmo.py](../python/addons/featurizer_elmo.py). The `featurizer_type` should be passed as an argument as shown below:

```
python tag.py --input elmo-tester.conll --output elmo-tester-out.conll --model mead/elmo-2/tagger-model-tf-2658 --mxlen 60 --mxwlen 40 --model_type elmo --featurizer_type elmo
```

We support loading tagger models defined in [TensorFlow](../python/baseline/tf/tagger/model.py) and [PyTorch](../python/baseline/pytorch/tagger/model.py). 
