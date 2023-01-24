#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

cd $SUITESPARSE_PATH

# Uncompress tar file
for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
    rm "$f"
done

# Remove extra matrix info
for f in *.tar.gz.1; do
    rm "$f"
done

while read line; do
	rm "${line}_*.mtx"
done <$1

# Change file permissions of .mtx file
for f in *.mtx; do
    chmod ugo+r "$f"
done
