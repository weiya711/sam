#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

cd /nobackup/owhsu/sparse-datasets/suitesparse 

for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
    rm "$f"
done

for f in *.mtx; do
    chmod ugo+r "$f"
done
