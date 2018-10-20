## Reporting of Training Jobs

A `reporting hook` for baseline is a class that has three methods: `init`, `step` and `done`. The `step` method determines how the metrics in the training/validation/test step will be stored. For example, The `ConsoleReporting` prints out the results on the console.

You can use the reporting hooks in two ways:

- **mead config file**: The config file should have a field called "reporting". Eg:

```json
 "train": {
	"epochs": 2,
	"optim": "adadelta",
	"eta": 1.0,
	"model_zip": true,
	"early_stopping_metric": "acc",
      "verbose": {"console": true, "file": "sst2-cm.csv"}
    },
    "reporting":["visdom","xpctl"]
}
```

- pass in the reporting hooks as arguments of `mead-train` or `python/mead/trainer.py`: `mead-train --config sst2.json --reporting visdom console`

Following reporting hooks are implemented:

- **console**: prints results to the console.
- **log** (default): saves results to a log file.
- **visdom** : shows graphs in the visdom interface.
- **tensorboard**: shows graphs in the tensorboard interface.
- **xpctl**: stores the results in an [xpctl](xpctl.md) database.

Each reporting hook can be instantiated with their own parameters. These parameters can be specified in `python/mead/config/mead-settings.json`. For example, the `xpctl` reporting hook requires the credentials for the database and whether the model checkpoint files are to be saved or not. The `visdom` reporting hook can optionally take the name of the environment. The following is an example `python/mead/config/mead-settings.json`:

If the reporting hook is a user defined add-on it is convenient to include a `module` field that is the name of the python module the hook is defined in. Otherwise the module must be included in the `modules` section of the mead config for the hook to be loaded.

```json
{
 "datacache": "<path to home>/.bl-data",
  "reporting_hooks":{
    "visdom": {
      "name": "test"
    },
    "xpctl": {
      "cred": "<path to the xpctl cred file>",
      "save_model": true,
      "module": "reporting_xpctl"
    }
  }
}

```

You can also change/add these parameters during run time, eg:, the following command `mead-train --config sst2.json --visdom:name main` will change the  _name_ parameter for `visdom` from _test_ to _main_. You need to use namespacing for this: all parameters to be used by the `visdom` hook should start with `visdom:`.

Following `baseline`'s design philosophy, you can always write your own reporting hooks and put them in a place accessible by your `PYTHONPATH`. These files should be named like `reporting_x.py` where x=name of the hook. An example can be found at [`reporting_xpctl.py`](../python/addons/reporting_xpctl.py) which provides a reporting hook for `xpctl`.

### Using visdom

Here are step-by-step instructions for seeing training progress with visdom:

-  **Install visdom**: `pip install visdom`

-  **Launch visdom**: `python -m visdom.server` at any directory.

-  Navigate your browser to: http://localhost:8097/

-  Run mead job with visdom argument: either `mead-train --config <config.json> --reporting visdom` or `mead-train --config <config.json>` where the config contains `reporting: ["visdom"]`.

-  Losses will plot as the training progresses. You can compare runs by using the `visdom:name` argument in `mead-train` or changing `name` in `mead-settings`.


### Using tensorboard

*TODO*


#### Source of Truth for Reporting Hooks

There are a few places in mead that reporting hooks can be configured:

 * In mead-settings there is a section called `reporting_hooks` This is the default values for setting on the reporting hooks. Values here will always be included in config unless overridden.
 * In mead-config there is a section called `reporting` this is a list of active hooks. Hook options cannot be defined here.
 * From the command lines the `--reporting` flag is a list of active hooks. Command line arguments of the form `--hook:option value` will overwrite the default values in the mead-settings.

The main take-away is that `--reporting` and the `reporting` section of mead config are used to decide which hooks are active. `reporting_hooks` and `--hook:option value` args are used to set parameters of the hook with command line arguments overriding the defaults in mead-settings.
