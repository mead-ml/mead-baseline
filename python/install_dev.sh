#!/bin/bash

package=${1:-"baseline"}
test=${2:-"test"}

if [ $package = "baseline" ]; then
    EGG=xpctl.egg-info
else
    EGG=deep_baseline.egg-info
fi

clean_up() {
    rm -rf setup.py &> /dev/null
    mv "$EGG.old" $EGG &> /dev/null
    rm -rf README.md &> /dev/null
    rm -rf MANIFEST.in &> /dev/null
}
trap clean_up EXIT ERR INT TERM

mv $EGG "$EGG.old" &> /dev/null

cp setup_$package.py setup.py
if [ $? != 0 ]; then
    echo "No setup file for $package was found, file should be named setup_$package.py"
    exit 1
fi

pip install -e .[test,sql,mongo]
if [ $? != 0 ]; then
    echo "$package failed to install."
    exit 1
fi

if [ $package = "baseline" ]; then
    if [ $test = "test" ]; then
        pytest
    fi
fi
