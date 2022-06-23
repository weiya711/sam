#!/bin/bash

for numpy in $1/*-numpy.tns; do
    taco=${numpy/-numpy/-taco}
    if [ ! "$(wc -l < $numpy | xargs)" -eq "$(wc -l < $taco | xargs)" ]; then
        echo "Files $numpy and $taco have a differing number of entries."
    fi
done
