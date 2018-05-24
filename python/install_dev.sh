#!/bin/bash

package=${1:-"baseline"}
test=${2:-"test"}

if [ $package = "baseline" ]; then
    EGG=xpctl.egg-info
else
    EGG=deep_baseline.egg-info
fi

mv $EGG "$EGG.old"

cp setup_$package.py setup.py
pip install -e .[test]
rm setup.py

mv "$EGG.old" $EGG

if [ $package = "baseline" ]; then
    if [ $test = "test" ]; then
        pytest
    fi
fi

rm -rf README.md
rm -rf MANIFEST.in
