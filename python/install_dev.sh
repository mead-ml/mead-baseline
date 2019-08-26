#!/bin/bash

test=${1:-"test"}

clean_up() {
    rm -rf README.md &> /dev/null
    rm -rf MANIFEST.in &> /dev/null
}
trap clean_up EXIT ERR INT TERM

pip install -e .[test,yaml]

if [ $? != 0 ]; then
    echo "$package failed to install."
    exit 1
fi

if [ $test = "test" ]; then
    pytest --forked
fi
