#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

BENCHMARKS=(
  tensor3_innerprod_FINAL
  tensor3_elemadd_FINAL
  tensor3_ttv_FINAL
  tensor3_ttm_FINAL
  tensor3_mttkrp_FINAL
)

TENSORS=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)


errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

# LANKA
if [ $1 -eq 1 ]; then
	export SUITESPARSE_PATH=/data/scratch/changwan/florida_all
	export FROSTT_PATH=/data/scratch/owhsu/datasets/frostt
	export TACO_TENSOR_PATH=/data/scratch/owhsu/datasets
	export SUITESPARSE_FORMATTED_PATH=/data/scratch/owhsu/datasets/suitesparse-formatted
	export FROSTT_FORMATTED_TACO_PATH=/data/scratch/owhsu/datasets/frostt-formatted/taco-tensor
	export FROSTT_FORMATTED_PATH=/data/scratch/owhsu/datasets/frostt-formatted
	
	mkdir -p $TACO_TENSOR_PATH
	mkdir -p $SUITESPARSE_FORMATTED_PATH
	mkdir -p $FROSTT_FORMATTED_TACO_PATH
	mkdir -p $FROSTT_FORMATTED_PATH

	lanka=ON
	neva=OFF
elif [ $1 -eq 2 ]; then
	lanka=OFF
	neva=ON
else
	lanka=OFF
	neva=OFF
fi

format_outdir=${FROSTT_FORMATTED_PATH} 
basedir=$(pwd)
frosttpath=$FROSTT_PATH
benchout=frostt-bench/sam

__conda_setup="$('/data/scratch/owhsu/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/scratch/owhsu/miniconda/etc/profile.d/conda.sh" ]; then
        . "/data/scratch/owhsu/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/data/scratch/owhsu/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate aha

mkdir -p "$benchout"
mkdir -p $format_outdir
mkdir -p $TACO_TENSOR_PATH/other-formatted-taco

make -j8 taco/build NEVA=$neva LANKA=$lanka GEN=ON

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	for t in ${!TENSORS[@]}; do
		name=${TENSORS[$t]}
		cd $format_outdir


		if [ "$bench" == "tensor3_innerprod_FINAL" ]; then
			echo "Generating input format files for $name..."

			$basedir/compiler/taco/build/bin/taco-test sam.pack_sss012
			$basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt
			python $basedir/scripts/datastructure_frostt.py -n $name -f sss012
			python $basedir/scripts/datastructure_frostt.py -n $name -f sss012 --other
			chmod -R 775 $FROSTT_FORMATTED_PATH
		fi

		cd $basedir/sam/sim

		pytest test/final-apps/test_$bench.py --frosttname $name --benchmark-json=$path/$name.json 
		python $basedir/scripts/converter.py --json_name $path/$name.json	
		    
		status=$?
		if [ $status -gt 0 ]
		then 
		  errors+=("${name}, ${bench}")
		fi

		cd $basedir
	done 

	python $basedir/scripts/bench_csv_aggregator.py $path $basedir/$benchout/frostt_$bench.csv

	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done

