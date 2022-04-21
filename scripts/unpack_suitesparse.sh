#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

cd data/suitesparse/

for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
done
