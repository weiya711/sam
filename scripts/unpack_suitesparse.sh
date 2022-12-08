#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

cd $SUITESPARSE_PATH

for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
    rm "$f"
done

for f in *.mtx; do
    chmod ugo+r "$f"
done
