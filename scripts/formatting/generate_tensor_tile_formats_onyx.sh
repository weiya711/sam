#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/generate_frostt_formats_onyx.sh <tensor_name.txt>

shopt -s extglob

FORMATS=(
  sss012
)

# using all tensor apps except elemmul here**
BENCHMARKS=(
  # tensor3_elemadd
  # tensor3_innerprod
  tensor3_ttv
  # tensor3_ttm
  # tensor3_mttkrp
  # tensor3_elemmul
  # tensor3_mttkrp
)

OTHER_FORMATS=(
  # sss
  # sss
  s
  # ss
  # ss
  # sss
  # ss
	)

OTHERBENCHES='["tensor3_ttv", "tensor3_ttm", "tensor3_mttkrp"]'

basedir=$(pwd)

old_frostt_formatted_taco_path=$FROSTT_FORMATTED_TACO_PATH
old_frostt_path=$FROSTT_PATH
old_frostt_tensor_path=$FROSTT_TENSOR_PATH
old_frostt_formatted_path=$FROSTT_FORMATTED_PATH

set_temp_env(){
	export FROSTT_FORMATTED_TACO_PATH=$1;
	export FROSTT_PATH=$2 
	export FROSTT_TENSOR_PATH=$3; 
	export FROSTT_FORMATTED_PATH=$4; 
}

run_format(){
	f=$1
	bench=$2
	format=$3
	bench_path=$4
	tile_path=$5
	tensor_format=$6

	basedir=$(pwd)
	echo "Processing $f"
	filename=$(basename $f)
	name=${filename%.*}
	echo "Processing $name"

	export FROSTT_FORMATTED_TACO_PATH=$bench_path/taco/ 
	export FROSTT_PATH=$tile_path 
	export FROSTT_TENSOR_PATH=$filename 
	export FROSTT_FORMATTED_PATH=$bench_path/formatted/
	export TENSOR_FORMAT=$tensor_format
    echo "Tensor format: $TENSOR_FORMAT"

	$basedir/compiler/taco/build/bin/taco-test sam.pack_$format

	python $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format --other -b $bench -hw --output_dir $bench_path/formatted/$name
    
    echo "Done processing $name" 
}

export -f run_format

for i in ${!FORMATS[@]}; do
    format=${FORMATS[@]};
    echo "Generating files for format $format..."

    for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	other_format=${OTHER_FORMATS[$b]}

	while read line; do
    
		bench_path=$basedir/tiles/$bench/$line
		mkdir -p $bench_path 
		tile_path=$bench_path/mtx
		
		echo "Tiling tns file"
		python $basedir/sam/sim/src/tiling/tile.py --tensor_type frostt --input_tensor $line --cotile $bench --multilevel --hw_config $2 --higher_order --output_dir_path $bench_path

		mkdir -p $bench_path/taco
        mkdir -p $bench_path/formatted/$name


	 	find $tile_path -name tensor_B_tile_*.tns -maxdepth 1 | parallel run_format {} $bench $format $bench_path $tile_path "sss" 
		
		find $tile_path -name tensor_*_tile_*.tns -not -name tensor_B_tile_*.tns -maxdepth 1 | parallel run_format {} $bench $format $bench_path $tile_path "$other_format" 

	done < $1
        
	set_temp_env $old_frostt_formatted_taco_path $old_frostt_path $old_frostt_tensor_path $old_frostt_formatted_path 
        chmod -R 775 $FROSTT_FORMATTED_PATH
    done
done
