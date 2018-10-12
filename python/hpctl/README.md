# HyperParameter Optimization Control (hpctl)

Automatic Hyper Parameter Optimization to use with baseline, MEAD, and xpctl. Multiple configurations are sampled with random or grid based sampling methods. Parallel jobs are scheduled and results are aggregated and displayed.

Jobs are identified with a label that has 3 parts:

 * The experiment hash: This is the hash of the hpctl configuration file (with the sampling directives)
 * The run hash: The hash of the MEAD config (hpctl config after sampling) for this run.
 * Random name: A random name formed from two adjectives and one noun. This is used to separated jobs with the same config used for `hpctl verify`.

## `hpctl search`

Search over a given configuration file. This version runs jobs drawn from a single hpctl experiment (a single configuration file).

This has the normal mead arguments as well as extra `--hpctl-logging` which is the file that has configuration information for the hpctl logging (post and hostname for the server that collects logs.)

## `hpctl verify`

Used to test the variance of hyperparameters, designed to run the same configuration many times and automatically save all results using xpctl.

## `hpctl serve`

Used to launch a hpctl server where jobs are submitted to it and it runs them. This takes a settings file used to configure the backend, a embeddings index, and dataset index. Exposes a flask frontend to get information about running jobs. This mode can handle jobs from multiple experiments (generated from different config files).

## `hpctl launch`

Used to launch a one-off job to the hpctl server.

## `hpctl list`

This lists information about all the hpctl jobs for all the experiments.

## `hpctl find`

This finds the location on the file system for a single job by name.

## xpctl

hpctl has two ways it interacts with xpctl. The first is manually where hpctl presents the user the choice to save given runs to xpctl or not. The second is automatically where each job run with MEAD is configured to use the xpctl reporting hook to automatically save the results to xpctl. If xpctl configuration is found in the settings file then the manual option is automatically activated. If `--reporting xpctl` is passed via the command line then the automatic version is used.

## Local Vs Remote

hpctl has two different modes, `local` mode where hpctl is a single application and all pieces run on your computer and `remote` mode where it is run as a client server application. Objects have two implementations, local is run when using it local (it is also how the server uses them) and remote which makes http requests to the server. Local and Remote mode (controlled by the backend type in the settings file) allows a user to run the same `hpctl search` command and not know if jobs are running on a local node or the server.

## Backends

Currently hpctl supports several backends to run jobs:

 * `mp`: Using native python multiprocessing to launch and control jobs.
 * `docker`: Using docker to launch isolated jobs.
 * `remote`: Send the settings for a job to a remote server to run. The server then runs jobs with a different backend.

These backends manage GPUs themselves. They are given a list of gpus they are allowed to use and they give them to job using environment variables.

We plan to add support for backends with better global gpu schedulers and that allow for running jobs across nodes such as slurm or k8s.

Configuration for the backend can also be overridden from the command line using `--backend:option value`.

## Scheduling

When running as a server hpctl must queue jobs when they come in but no GPUs are available. The schedule is currently simple, it can be thought of as a Round Robin Scheduler from OSs. Several programs (experiments from users) are competing for a single resource (the backend) and are selected in a round robin manor. This version the scheduler doesn't do things like take into considerations how many gpus a jobs needs.

## Frontends

hpctl provides some simple frontends out of the box, one that prints to the console and updates with every training event, one that prints to console that only shows the best valid set scores. There is also a flask frontend that is used for the server mode.

Like the backend options are defined in the settings file and can be overridden from the command line with `--frontend:options value`

## Logging

Logging is performed with json logs over the python socket logging functionality. The logging config should include the port for the socket as well as the host name for a server that collects these logs.

The logging server is automatically started and stopped so the user doesn't need to worry about it.

## hpctl Settings.

These are possible settings you can have in the `hpctl` section of the mead settings.

 * `backend`
   * `type`: The backend to use `{mp, docker, remote}`
   * `real_gpus`: The index of local GPUs hpctl is allowed to use.
   * `default_mounts`: A list of location to mount `ro` in docker.
 * `logging`
   * `host`: The location of the server that collects logs.
   * `port`: The port log to.
 * `frontend`
   * `type`: The type of frontend to use `{console, console_dev, flask}`
   * `train`: The training statistic to print.
   * `dev`: The dev set stat to print (and select best on).
   * `test`: The dev set stat to print.


## ConfigSampler

hpctl configuration files are MEAD configurations file plus sampling directives.  In the following example a simple model config is augmented to perform a hyper-parameter search. The size of the hidden layers (`hsz`) is randomly drawn from `150, 200, 150` and the dropout value is drawn from a normal distribution centered at `0.5` with a spread of standard deviation of `0.1`.

```json
  "model": {
    "model_type": "lstm",
    "hsz": 200,
    "dropout": 0.5
  }
```

```json
  "model": {
    "model_type": "lstm",
    "hsz": {
      "hpctl": "choice",
      "values": [150, 200, 250]
    },
    "dropout": {
      "hpctl": "normal",
      "mu": 0.5,
      "sigma": 0.1
    }
  }
```

### Samplers

There are several provided samplers as well as the ability to provide new ones. A sampling directive is a dictionary that replaces the parameter value that always includes the key `"hpctl"` and the name of the sampling procedure is the value. The other keys and values are the arguments and parameters to the sampling function.

Included samplers and their parameters:

 * `normal(mu, sigma)`
 * `uniform(min, max)`
 * `min_log(min, max)` Sample on a log scale so that more values are drawn closer to the min value.
 * `max_log(min, max)` Sample on a log scale but skew towards the max.
 * `uniform_int(min, max)` Sampler integers between min and max inclusive.
 * `choice(values)` values is a list of values that choice draws from.
 * `grid(values)` values is a list of values. All parameters that use grid search are aggregated together into a single grid.

#### Addon Samplers

__UPDATE WHEN NEW REG IS MERGED__ When reading a configuration file hpctl first finds all the types of samplers. Anything value for the `"hpctl"` key that is not in the default samplers is assumed to be a user sampler. These are user defined sampling classes. They should be defined in a file called `sampler_XXXX` where `XXXX` is the value of the `"hpctl"` key. There needs to be `create_sampler` function.

The class should expose 3 things:

 1) A `.name` that is `XXXX`
 2) A `.adder` which is a callable that takes a dictionary, key, and a dictionary. This second dictionary is the sampling directive and the functions should know how to parse the data in it and saves it into the first dict with the key. The output of this is saved to the `.values` property on the object.
 3) A `.sample` that produces a new value for the parameters. This returns a dictionary of keys to sampled values.


### Constraints

Sampling also supports the ability to put constraints on the values sampled. This is useful for things like sampling for `eta` because using a negative learning rate doesn't make sense (some frameworks like pytorch this causes an error).

Constraints are supplied in the sampling directive dictionary with the `constraints` key. The value is a list of constraints. Constraints are a string in the form `op value`. For example `> 0` or `<= 1`. Conceptually the constraint is always "sampled" "compared to" "value". Multiple constraints are `ANDed` together.

Constraints use the python `eval` function and can be some what complex. For example to insure sampled sizes for a bLSTM allow splitting between a forward and backward LSTM the constraint `% 2 == 0` can be used.

Note: Constraints are not supported for sampling methods where the values are enumerated by the user (choice and grid).
