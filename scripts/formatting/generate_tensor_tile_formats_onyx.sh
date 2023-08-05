#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/generate_frostt_formats_onyx.sh <tensor_name.txt>

shopt -s extglob

FORMATS=(
  sss012
)

BENCHMARKS=(
  # using all tensor apps except elemmul here**
  tensor3_elemadd
  # tensor3_innerprod
#   tensor3_ttv
  # tensor3_ttm
  # tensor3_mttkrp
  # tensor3_elemmul
  # tensor3_mttkrp
  # tensor3_ttm
)

OTHERBENCHES='["tensor3_ttv", "tensor3_ttm", "tensor3_mttkrp"]'

basedir=$(pwd)

for i in ${!FORMATS[@]}; do
    format=${FORMATS[@]};
    echo "Generating files for format $format..."

    for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
    
	bench_path=$basedir/tiles/$bench
	# tile_path=$bench_path/mtx
	#NOTE: had to rename from mtx to tns here manually - should find what causes it to be mtx in first place/switch to tns
	tile_path=$bench_path/tns
	mkdir -p $bench_path/taco

	for f in $tile_path/tensor_B_*; do
		echo "Processing $f"
		filename=$(basename $f)
		name=${filename%.*}
		echo "Processing $name"
		FROSTT_FORMATTED_PATH=$bench_path/taco FROSTT_PATH=$tile_path FROSTT_TENSOR_PATH=$filename $basedir/compiler/taco/build/bin/taco-test sam.pack_$format
		FROSTT_FORMATTED_PATH=$bench_path/taco FROSTT_PATH=$tile_path FROSTT_TENSOR_PATH=$filename python $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format --other -b $bench -hw
	done
	
	for f in $tile_path/!(tensor_B_*); do 
		echo "Processing $f"
		filename=$(basename $f)
		name=${filename%.*}
		FROSTT_FORMATTED_PATH=$bench_path/taco FROSTT_PATH=$tile_path FROSTT_TENSOR_PATH=$filename $basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt
		FROSTT_FORMATTED_PATH=$bench_path/taco FROSTT_PATH=$tile_path FROSTT_TENSOR_PATH=$filename python $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format --other -b $bench -hw
	done
        
        chmod -R 775 $FROSTT_FORMATTED_PATH
    done
done
