# HyperParameter Optimization Control (hpctl)

Automatic Hyper Parameter Optimization to use with baseline.

Supports multiple backends (multiprocessing, k8s)

Supports Random, Grid, and hybrid search for optimal hyperparameters.

Collects performance from launched jobs using python network based logging.

Optional Dependencies: `setproctitle`


## HPCTL settings.

These are possible settings you can have in the `hpctl` section of the mead settings.

 * `backend`
   * `type`: The backend to use `{mp, docker}`
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
