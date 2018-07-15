### Datasets and Embedding Files

The datasets and embedding file locations should be provided in `mead/datasets.json`, and `mead/emeddings.json`. These files can exist in your local machine. We also provide methods for automated download. There are a couple of ways to specify the dataset or embedding file locations:

#### Requirements

`requests` is the dependency for automatic download of the files. This can be done in most environments with the command `pip install requests`.
 
#### Datasets

- Path to the data files on your computer (provide the paths separately):

```
    {
      "train_file": "/data/datasets/ner/wnut-gaz/wnut17train.conll",
      "valid_file": "/data/datasets/ner/wnut-gaz/wnut17dev.conll",
      "test_file": "/data/datasets/ner/wnut-gaz/wnut17test.conll",
      "label": "wnut-gaz"
    }
```

- Link to a directory zip, file names in the unzipped directory as keys (we provide _sha1_ for the zip file in this case. If you send a PR for a new dataset, please add the _sha1_ as well)


```
 {
    "train_file": "eng.train",
    "valid_file": "eng.testa",
    "test_file": "eng.testb",
    "download": "https://www.dropbox.com/s/p6ogzhiex9yqsmn/conll.tar.gz?dl=1",
    "sha1":"521c44052a51699742cc63e39db514528e9c2640",
    "label": "conll"
  }
```

or

```
{
    "vocab_file": "vocab.en_vi",
    "train_file": "train",
    "valid_file": "tst2012",
    "test_file": "tst2013",
    "download": "https://www.dropbox.com/s/99petw2kdab69cr/iwslt15-en-vi.tar.gz?dl=1",
    "sha1":"418a2ccfa7c46a1a1db900295e95ac03e2ec6993",
    "label": "iwslt15-en-vi"
  }
```

- Each key points to a separate download link:

```
{
    "train_file": "https://www.dropbox.com/s/1jxd5tpc6lo12t7/train-tok-nodev.txt.gz?dl=1",
    "valid_file": "https://www.dropbox.com/s/8wgf8m6f8qmubga/dev-tok.txt.gz?dl=1",
    "test_file": "https://www.dropbox.com/s/3qf0sj60exi4r1r/test-tok.txt.gz?dl=1",
    "label": "dbpedia"
}
```

#### Embedding Files

- Compressed zip/tar.gz file with multiple embedding files. the files are uniquely identified by the dsz. Eg, in the zip below, we will pick the file `200` in the file name.

```
    {
	"label": "glove-twitter-27B",
	"file": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
	"dsz": 200
    },
```

- Direct download link

```
    {
        "label": "glove-6B-50",
        "file": "https://www.dropbox.com/s/339mhx40t3q9bp5/glove.6B.50d.txt.gz?dl=1",
        "dsz": 50
    },
```

- On file system

```
{
        "label": "glove-6B-100",
        "file": "/data/embeddings/glove.6B.100d.txt",
        "dsz": 100
    },
```

### File format
The links can have the usual data format supported by `baseline` or standard zip formats such as `.gz, tar.gz, tgz, zip`. We automatically extract them as needed.

#### Caching

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


#### Writing own downloaders

You can write your own downloaders by extending the base Downloader class. Helper methods are provided.

