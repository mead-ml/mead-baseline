## Visualization of Training Jobs

### Using visdom

Here are step-by-step instructions for seeing training progress with visdom:

#### Install visdom
```
pip install visdom
```

#### Launch visdom

```
dpressel@dpressel:~/dev/work/baseline/python/mead$ python -m visdom.server
```
Now, navigate your browser to: http://localhost:8097/


#### Run mead job with visdom argument

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

Losses will plot as the training progresses.  You can compare runs by using the `visdom_name` argument in the config.

```
dpressel@dpressel:~/dev/work$ head -n 9 mead/config/visdom_name_config.json
{
    "batchsz": 50,
    "visdom": true,
    "visdom": "example",
    "preproc": {
	"mxlen": 100,
	"rev": false
    },
    "backend": "tensorflow",
```

### Using tensorboard

*TODO*
