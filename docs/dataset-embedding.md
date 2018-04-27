### Datasets and Embedding files

The datasets and embedding file locations are tracked in `mead/datasets.json`, and `mead/emeddings.json`. The datasets can exist in your local machine. We also provide methods for automated download of the embedding and the data files. There are a couple of ways to do this:
 
#### Datasets
 
 In `datasets.json`. 

- Provide direct download links for train/test/valid keys:
```aidl
{
    "train_file": "https://www.dropbox.com/s/a3mg6nuk1u9wbk1/train-tok-nodev.txt.gz?dl=1",
    "valid_file": "https://www.dropbox.com/s/0f3zw5okduxtcjg/dev-tok.txt.gz?dl=1",
    "test_file": "https://www.dropbox.com/s/n5rbul7ku2ek6zn/test-tok.txt.gz?dl=1",
    "label": "dbpedia"
  }
```
- Provide download link to a zip file that will extract into a folder, and specify the filenames inside the folder. 

```aidl
  {
    "train_file": "eng.train",
    "valid_file": "eng.testa",
    "test_file": "eng.testb",
    "download": "https://www.dropbox.com/s/35pfeppg7n5yg6p/conll.tar.gz?dl=1",
    "label": "conll"
  }
```

#### Embedding files

You need to provide direct links to the embedding files:
 
```aidl
  {
        "label": "glove-6B-100",
        "file": "https://www.dropbox.com/s/cjg716n67rpp9s5/glove.6B.100d.txt.gz?dl=1",
        "dsz": 100
    },
```

### File format
The links can have the usual data format supported by `baseline` or standard zip formats such as `gz, tar.gz, tgz, zip`. We automatically extract them as needed.

#### Caching

For faster download, all downloaded files are cached. A `<key,value>` store for the download links are maintained at an internal JSON file, which should not be committed. For eg:
```aidl
x:config$ cat datasets-embeddings-cache.json 
{
 "https://www.dropbox.com/s/p6ogzhiex9yqsmn/conll.tar.gz?dl=1": "/data/bl-dataset-embeddings//521c44052a51699742cc63e39db514528e9c2640",
 "https://www.dropbox.com/s/cjg716n67rpp9s5/glove.6B.100d.txt.gz?dl=1": "/data/bl-dataset-embeddings//a483a44d4414a18c7b10b36dd6daa59195eb292b",
 "https://www.dropbox.com/s/sj9xjeiihjs8cmk/oct27.train?dl=1": "/data/bl-dataset-embeddings//12de099c6bc7d1f10a50afcd0bbc004e902aa759",
 "https://www.dropbox.com/s/whzkv7te2zklqn2/oct27.dev?dl=1": "/data/bl-dataset-embeddings//bb5833a28d4c824342068a68a55fe984bd3155b8",
 "https://www.dropbox.com/s/riyn2ne85pirfpd/oct27.test?dl=1": "/data/bl-dataset-embeddings//45f79cbf1bdb98db6999e647c2d5a7ec41e4dced",
 "http://nlp.stanford.edu/data/glove.twitter.27B.zip": "/data/bl-dataset-embeddings//dce69c404025a8312c323197347695e81fd529fc"
}

```


The location of the cache directory is `~/.bl-dataset-embeddings/` by default, unless you explicitly mention it at `mead/config.json`. 


 
#### Writing own downloaders

You can write your own downloaders by extending the base Downloader class. Helper methods are provided. 

 
