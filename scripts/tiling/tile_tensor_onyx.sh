#!/bin/bash

# THIS FILE MUST BE RUN FROM sam/ location
# ./scripts/tiling/tile_ext.sh <tile_dir> <arch_config.yaml>

BENCHMARKS=(
#   matmul_ikj
	# mat_mattransmul
	tensor3_elemadd
)

sspath=$SUITESPARSE_PATH
frosttpath=$FROSTT_PATH

basedir=$(pwd)

# tiles_path=$basedir/extensor_mtx/$1
tiles_path=$basedir/extensor_tns/$1

echo "tiles_path is: $tiles_path"

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	rm -rf $basedir/tiles/$bench/*

	while read line; do 
		echo "i got here!"
		# echo "Tiling mtx file"
		echo "Tiling tns file"
		python ./sam/sim/src/tiling/tile.py --tensor_type frostt --input_tensor $line --cotile $bench --multilevel --hw_config ./sam/sim/src/tiling/memory_config_onyx.yaml --higher_order
	done <$1

done
