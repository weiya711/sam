#!/bin/bash

BENCHMARKS=(
#   matmul_ijk
#   mat_elemadd
	# mat_elemmul
	mat_mattransmul
)

# THIS FILE MUST BE RUN FROM sam/ location
sspath=$SUITESPARSE_PATH

basedir=$(pwd)

ext_path=$basedir/extensor_mtx/$1

echo "$ext_path"

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	rm -rf $basedir/tiles/*

	echo "Tiling mtx file"
	python3 $basedir/sam/sim/src/tiling/tile.py --extensor --input_path $ext_path --cotile $bench --multilevel --hw_config $basedir/sam/sim/src/tiling/$2 

	echo "Generating input format files for $ext_path..."
	python3 $basedir/scripts/datastructure_suitesparse.py -n temp -hw -b $bench --input $basedir/tiles/$bench/mtx/ --output_dir_path $basedir/tiles/$bench/formatted --tiles

done

