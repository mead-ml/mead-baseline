# Datasets, Embeddings and Vectorizers in MEAD

This section describes the index files MEAD uses to load datasets, embeddings and vectorizers, as well as details on the API and architecture design for the Embeddings and Vectorizer objects and how to create and use your own from MEAD.

## MEAD Indices

The driver programs use indices to map a unique identifier to a resource.  There are 3 types of resources that MEAD tracks with indices:

- datasets: mappings of a unique key to a downloadable or stored dataset
- embeddings: mappings of a unique key to some downloadable or stored embeddings model
- vecs: mappings of a unique key to a vectorizer model

Indices by default are in JSON, but if PyYAML is installed, they can also be in YAML format.

By default, the trainer uses the installed locations for the indices (`mead/config/datasets.json`, `mead/config/embeddings.json` and `mead/config/vecs.json`), but these locations can be overridden at the command-line.  They also do no have to be on the local filesystem -- the user may pass in a URL reference and the index will then be downloaded to the local system.  Additionally, there are set of existing indices that can be found at [mead-hub](https://github.com/mead-ml/hub).  MEAD provides a shorthand way of referencing mead-hub indices so you dont have to type in the full URL:

- embeddings: `hub:v1:embeddings`
- vecs: `hub:v1:vecs`

So as an example, to refer to the hub embeddings and vectorizers in your training, you can override them like this:
```
mead-train --embeddings hub:v1:embeddings --config config/sst2-bert-base-uncased.yml --vecs hub:v1:vecs
```
 
### Dataset Index Files

- Path to the data files on your computer (provide the paths separately):

```
[
    {
      "train_file": "/data/datasets/ner/wnut/wnut17train.conll",
      "valid_file": "/data/datasets/ner/wnut/wnut17dev.conll",
      "test_file": "/data/datasets/ner/wnut/wnut17test.conll",
      "label": "wnut"
    },
    ...
]
```
The YAML equivalent would be
```
- train_file: /data/datasets/ner/wnut17train.conll
  valid_file: /data/datasets/ner/wnut17dev.conll
  test_file" /data/datasets/ner/wnut17test.conll
  label: wnut
```

Locations can also be a link to a zipped directory, with file names in the unzipped directory as keys (we provide _sha1_ for the zip file in this case. If you send a PR for a new dataset, please add the _sha1_ as well)

```
[
 {
    "train_file": "eng.train",
    "valid_file": "eng.testa",
    "test_file": "eng.testb",
    "download": "https://www.dropbox.com/s/p6ogzhiex9yqsmn/conll.tar.gz?dl=1",
    "sha1":"521c44052a51699742cc63e39db514528e9c2640",
    "label": "conll"
  },
  {
    "vocab_file": "vocab.en_vi",
    "train_file": "train",
    "valid_file": "tst2012",
    "test_file": "tst2013",
    "download": "https://www.dropbox.com/s/99petw2kdab69cr/iwslt15-en-vi.tar.gz?dl=1",
    "sha1":"418a2ccfa7c46a1a1db900295e95ac03e2ec6993",
    "label": "iwslt15-en-vi"
  },
  ...
]
```

Each file can also be downloaded individually via a separate download link:

```
{
    "train_file": "https://www.dropbox.com/s/1jxd5tpc6lo12t7/train-tok-nodev.txt.gz?dl=1",
    "valid_file": "https://www.dropbox.com/s/8wgf8m6f8qmubga/dev-tok.txt.gz?dl=1",
    "test_file": "https://www.dropbox.com/s/3qf0sj60exi4r1r/test-tok.txt.gz?dl=1",
    "label": "dbpedia"
}
```

### Embedding Index Files

The embeddings index supports downloading compressed zip/tar.gz file with multiple embedding files. the files are implicitly identified by the `dsz` field (representing the embedding output hidden unit size). Eg, in the zip below, we will pick the file `200` in the file name.

```
[
  {
    "label": "glove-twitter-27B",
    "file": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
    "dsz": 200
  },
  ...
]
```

We also support direct download links:

```
[
  {
    "label": "glove-6B-50",
    "file": "https://www.dropbox.com/s/339mhx40t3q9bp5/glove.6B.50d.txt.gz?dl=1",
    "dsz": 50
  },
  ...
]
```

Embeddings can also be referenced from the file system

```
[
  {
    "label": "glove-6B-100",
    "file": "/data/embeddings/glove.6B.100d.txt",
    "dsz": 100
  },
  ...
]
```

### File formats supported for download

The links can have the usual data format supported by `baseline` or standard zip formats such as `.gz, tar.gz, tgz, zip`.  The embeddings downloader code automatically checks the file header to see if its in an extractable format, and in most cases, we automatically extract them as needed.  If the embedding should not be extracted, the index entry should contain an  `unzip` property in the index is set to `false`.  In that case, we will not unpack the model.  This functionality is used for BERT NPZ checkpoints, for example, which are in the PKZIP format but which we dont want to unzip, since the checkpoint is opened by the model itself.

### Caching

For faster download, all downloaded files are cached. A `<key,value>` store for the download links are maintained at an internal JSON file (datasets-embeddings-cache.json), which should not be committed. For eg:
```
x:config$ cat data-cache.json
{
 "https://www.dropbox.com/s/p6ogzhiex9yqsmn/conll.tar.gz?dl=1": "/data/bl-dataset-embeddings//521c44052a51699742cc63e39db514528e9c2640",
 "https://www.dropbox.com/s/cjg716n67rpp9s5/glove.6B.100d.txt.gz?dl=1": "/data/bl-dataset-embeddings//a483a44d4414a18c7b10b36dd6daa59195eb292b",
 "https://www.dropbox.com/s/sj9xjeiihjs8cmk/oct27.train?dl=1": "/data/bl-dataset-embeddings//12de099c6bc7d1f10a50afcd0bbc004e902aa759",
 "https://www.dropbox.com/s/whzkv7te2zklqn2/oct27.dev?dl=1": "/data/bl-dataset-embeddings//bb5833a28d4c824342068a68a55fe984bd3155b8",
 "https://www.dropbox.com/s/riyn2ne85pirfpd/oct27.test?dl=1": "/data/bl-dataset-embeddings//45f79cbf1bdb98db6999e647c2d5a7ec41e4dced",
 "http://nlp.stanford.edu/data/glove.twitter.27B.zip": "/data/bl-dataset-embeddings//dce69c404025a8312c323197347695e81fd529fc"
}

```


The location of the cache directory is `~/.bl-data/` by default, unless you explicitly mention it at `mead/config/meadconfig.json` and pass it to the trainer with the option `--meadconfig`: `python trainer.py --config config/twpos.json --task tagger --meadconfig config.json`


### Writing your own downloaders

You can write your own downloaders by extending the base Downloader class. Helper methods are provided.

## API and Architecture Design


### Vectorizers API

The `Vectorizer` interface in mead accepts a `List[str]` as input and produces frequency dictionaries when the `count()` function is called, and `numpy.array`s when the `run()` function is called.

The purpose of `count()` is to tabulate the frequencies to produce a vocabulary.  The returned object is a `Counter` and may be filtered by low-frequency terms, or manipulated in any other way that makes sense.  While the `Vectorizer` knows how to `count()` tokens, it does not store anything internally.

When its time to execute code, the `run()` function is called on a `List[str]` input and an index that maps tokens to integers.  The vectorizer then produces a fixed dimensional `numpy.array` of `integers` that is typically zero-padded on the right side.  The `Vectorizer` interface is not expected to produce mini-batches of data, it is expected that the caller should be able to orchestrate the individual outputs of the `Vectorizer` into a proper batch.

How the caller orchestrates these indices is unknown to the `Vectorizer`.  It just knows how execute those functions.  A `Vectorizer` also can contain a `transform` function, which it runs over each token.  Typically the purpose of this function is simply to lower-case the input, but it can be used for any purpose (e.g. stop-wording).

`Vectorizer`s are currently persisted as `.pickle` files, which makes it easy to reload them in Python, but is quite limited for other languages, and this is likely to change in the near future.

`Vectorizer`s currently a one-to-one relationship with features.  This is almost always desirable as it provides better Separation of Concerns, but on rare occrasions, it can cause some complexities.

#### The dimensionality of Vectorizer output

For words or sub-words, usually the vectorizer will be 1D.  The return of `run()` would be a tensor padded out to `mxlen` and a scalar for that tensor's valid length.  Note that byte-pair (BPE) and WordPiece vectorizers are typically simple 1D vectorizers, but that in those cases, the output sequence length will usually be greater than the number of input tokens, because these tokenizers split words into sub-words.  Care must be taken to ensure that the `mxlen` is sufficient to fit the sub-word vectors or the results may be truncated.

To produce characters for each word, the vectorizer will typically be 2D, with `mxwlen` describing the length of each "word" and `mxlen` defining the overall sequence max length.  Another use-case for 2D vectorizers would be in Dialogue Act Recognition (DAR), where we may have a dimension for word-level tokenization for each turn, and a dimension for the turn itself.  In this case, `mxlen` is the maximum number of turns and `mxwlen` is the maximum turn length.

Sometimes models provide multiple features representing the same tokens.  This quite common in tagging, where the input file is typically in CONLL format, where each desired features is tab or space-delimited with a different feature identifier.  The `DictVectorizer` exists for these cases.  There are 1D and 2D representations of each feature when using `DictVectorizer`s.

One unusual and slightly complex use-case for `Vectorizers` occur when implementing taggers with BPE.  In those cases, typically we want our label vectorizer to produce "<PAD>" (zero-pad) features whenever the input is a non-first subword token.  But the only way to know that it would a sub-word is to first tokenize the surface tokens and test each to see if it is the start of a sub-word or not.  Because the problem arises during tagging, where our input is typically a CONLL file and we are using a dict based vectorizer, which has access to all the features including the surface terms.  In this case, we provide a special label dict vectorizer for BPE, as well as a surface version that simply reads and emits the surface and connect both of these to the reader.

#### Writing your own vectorizers

Here are some best practices for writing your own vectorizers
  - Determine if you only need a single representation of each token (or multiple representations via multiple embeddings).  If you want to use multiple pre-trained word embeddings with the same Vectorizer, this is possible by putting a list of labels in your mead-config for a single embedding.  If you need slightly different transforms of the same token, you should create separate list items defining a Vectorizer/Embedding pair.
  - Determine if you need to support multiple features per token or aggregate features (for instance `word-POS`. If so consider using a dictionary-style Vectorizer.  This is common for tagger use cases
  - Do you need to load a model in order to tokenize?  This is common for BPE and WordPiece.  If you are just using one of those, you can use our existing BPE or WordPiece vectorizers.  If not, you should load your vocab in the constructor

#### Embeddings API

TODO: update after this converges

#### How MEAD calls Embeddings and Vectorizers

Embeddings and Vectorizers are exposed to MEAD via `register_embeddings()` and `register_vectorizer()` respectively.  If you have your own object that you are providing and you wish for MEAD to call it, you should register your objects.

The core library provides a function to load embeddings `load_embeddings(name, **kwargs)`.  This function expects a previously loaded global registry of embeddings (named `MEAD_LAYERS_EMBEDDINGS`) with string keys and class values.

If your mead config specifies a `module` in its `embedding` section, this module will be dynamically added into the program (first it will be downloaded if the reference is  a URL) and will also be added dynamically to the `MEAD_LAYER_EMBEDDINGS` registry, making it immediately available to the function.


### Referring to your own Embeddings and Vectorizers

TODO:

### Calling objects from code

TODO:
