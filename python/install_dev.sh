#!/bin/bash

package=${1:-"baseline"}
test=${2:-"test"}

EGG=deep_baseline.egg-info


clean_up() {
    rm -rf setup.py &> /dev/null
    mv "$EGG.old" $EGG &> /dev/null
    rm -rf README.md &> /dev/null
    rm -rf MANIFEST.in &> /dev/null
}
trap clean_up EXIT ERR INT TERM

mv $EGG "$EGG.old" &> /dev/null

pip install -e .[test,yaml]

if [ $? != 0 ]; then
    echo "$package failed to install."
    exit 1
fi

if [ $package = "baseline" ]; then
    if [ $test = "test" ]; then
        pytest --forked
    fi
fi
