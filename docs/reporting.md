## Reporting of Training Jobs

A `reporting hook` for baseline is a class that has three methods: `init`, `step` and `done`. The `step` method determines how the metrics in the training/validation/test step will be stored. For example, The `ConsoleReporting` prints out the results on the console.

You can use the reporting hooks in two ways: 

- **mead config file**: The config file should have a field called "reporting". Eg: 

```
(dl) x:baseline$ tail -n 2 python/mead/config/sst2.json
    "reporting": ["visdom", "xpctl"]
}

```  

- pass in the reporting hooks as arguments of `mead-train` or `python/mead/trainer.py`: `mead-train --config sst2.json --reporting visdom, console` 

Following reporting hooks are implemented:

- **console**: prints results to the console.
- **log** (default): saves results to a log file.
- **visdom** : shows graphs in the visdom interface.
- **tensorboard**: shows graphs in the tensorboard interface.  
- **xpctl**: Stores results in an [xpctl](xpctl.md) database.

Each reporting hook can be instantiated with their own parameters. These parameters should be specified in `python/mead/config/mead-settings.json`. For example, the `xpctl` reporting hook requires the credentials for the database and whether the model checkpoint files are to be saved or not. The `visdom` reporting hook can optionally take the name of the environment. The following is an example `python/mead/config/mead-settings.json`:

```
{
 "datacache": "<path to home>/.bl-data",
  "reporting_hooks":{
    "visdom": {
      "name": "test"
    },
    "xpctl": {
      "cred": "<path to the xpctl cred file>",
      "save_model": true
    }
  }
}

``` 
 
#### FIXME: visdom behavior should be changed, also change the following README
 
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
