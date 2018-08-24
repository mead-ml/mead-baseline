## Visualization of Training Jobs

### Using visdom

Here are step-by-step instructions for seeing training progress with visdom:

### Install visdom
```
pip install visdom
```

### Launch visdom

```
dpressel@dpressel:~/dev/work/baseline/python/mead$ python -m visdom.server
```
Now, navigate your browser to: http://localhost:8097/


### Set up visdom logging
```
dpressel@dpressel:~/dev/work$ head -n 8 mead/config/mdconfig.json
{
    "batchsz": 50,
    "visdom": true,
    "preproc": {
	"mxlen": 100,
	"rev": false
    },
    "backend": "tensorflow",
```

Run mead job.  Losses will plot as the training progresses.  Make sure to clear your visdom display in between `mead-train` runs

### Using tensorboard

*TODO*