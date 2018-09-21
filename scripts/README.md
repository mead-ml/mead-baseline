# Scripts

A collection of tools for working with baseline.


### `bump.py`

A script to automatically bump version on baseline.

`python bump.py baseline dev`

This program bumps the dev version of baseline. The first argument is either a path to a `version.py` or one of `[baseline, xpctl, hpctl]`. These file should contain the version in the form `__version__ = Major.minor.patchdevX`.

It can bump the major, minor, patch, or dev version number. When bumping an version it reset all subsequent versions to 0.

There is also a `--test` option that displays the current version and the result of the bump but doesn't actually make the change.


### `speed_tests.py`

#### `python speed_tests.py run`

This runs a speed test and saves the results to a database.

 * `--config` Location of a config file or a directory of config files. If it is a directory it runs all configs inside it.
 * `--single` Run the config verbatim, don't rotate frameworks.
 * `--db` The database name to save results is (currently only sqlite)
 * `--trials` The number of times to run each config.
 * `--frameworks` A list of frameworks to run with this config.
 * It supports most of the `mead-train` arguments to find datasets and the like.

Running `python speed_tests.py run --config speed_configs/sst2.json --frameworks dynet pytorch` will run the sst2.json model with dynet and then pytorch.

#### `python speed_tests.py add`

This adds speed results to the database. This assume that you ran the model on the same environment as you are adding it from.

 * `--config` The config it was run with.
 * `--log` The location of the log file.
 * `--db` The database to save into.
 * `--gpu` The gpu to get info about.


#### `python speed_tests.py report`

Generate a speed test report from the database, It groups runs by things like config and framework version. It also only reports on the most recent software versions.

 * `--db` The database.
 * `--out` The name of the output file.

_Note: Report uses `pdflatex` and imagemagik `convert` to generate png files for the markdown report so this might fail if you don't have them installed._

#### `python speed_tests.py query`

Interface to pull specific information out of the database.

 * `--db` The database.
 * `--task` The task you want information about.
 * `--dataset` The dataset you want information about.
 * `--frameworks` The list of frameworks you want info about.
 * `--models` The models you want info about.

Frameworks and Models will be populated by the database is you don't supply them.


#### `python speed_tests.py explore`

Create the results table in the database so that you can query it easier.

 * `--db` The database.
