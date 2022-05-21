#!/usr/bin/env bash

# There are 3 ways to run this script:

# 1) execute without any arguments
# ./autoformatpy.sh
# applies isort, black and pyflakes to all *.py files returned by git ls-files

# 2) pass --staged flag
# ./autoformatpy.sh --staged
# applies isort, black and pyflakes to all staged (to be committed) *.py files

# 3) provide list of filepaths / patterns
# ./autoformatpy.sh filepath1.py filepath2.py prefix*py
# applies isort, black and pyflakes to the input files

cd `dirname $0` # set current working directory to the directory of this script

function apply_isort {
    local filepath=$1
    echo "applying isort to $filepath"
    isort $filepath
}

function apply_black {
    local filepath=$1
    echo "applying black to $filepath"
    black $filepath
}

function apply_pyflakes {
    local filepath=$1
    echo "applying pyflakes to $filepath"
    pyflakes $filepath
}

if [ $# -eq 0 ] # if this script was executed without any arguments
then
    # apply isort, black and pyflakes to all *.py files returned by git ls-files:
    git ls-files | grep .*\\.py | while read -r filepath ; do
        apply_isort $filepath
        apply_black $filepath
        # apply_pyflakes $filepath
    done
elif [ "$1" = "--staged" ] # if --staged flag was passed
then
    # apply isort, black and pyflakes to all staged (to be committed) *.py files:
    git diff HEAD --staged --name-only | grep .*\\.py | while read -r filepath ; do
        if [ -f $filepath ] # if file exists
        then
            apply_isort $filepath
            apply_black $filepath
            # apply_pyflakes $filepath
        fi
    done
else
    # apply isort, black and pyflakes to the input files:
    for filepath in "$@" # iterate over the input filepaths
    do
        apply_isort $filepath
        apply_black $filepath
        # apply_pyflakes $filepath
    done
fi
