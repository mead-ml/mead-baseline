# Scripts

A collection of tools for working with baseline.


### `bump.py`

A script to automatically bump version on baseline.

`python bump.py baseline dev`

This program bumps the dev version of baseline. The first argument is either a path to a `version.py` or one of `[baseline, xpctl, hpctl]`. These file should contain the version in the form `__version__ = Major.minor.patchdevX`.

It can bump the major, minor, patch, or dev version number. When bumping an version it reset all subsequent versions to 0.

There is also a `--test` option that displays the current version and the result of the bump but doesn't actually make the change.
