#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive
SECONDS=0
set -u

BENCHMARKS=(
matmul_ikj
matmul_ijk
)


NNZ=(
  5000
  10000
  25000
  50000
)

DIMENSIONS=(
 1024
 2360
 3696
 5032
 7704
 9040
 11712
 13048
 15720
)


errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

basedir=$(pwd)

export SAM_HOME=$basedir
benchout=memory_model_out
path=$basedir/$benchout
pushd .
mkdir -p "$benchout"

for nnz in ${!NNZ[@]}; do
	for dim in ${!DIMENSIONS[@]}; do
		if [ $2 -eq 1 ]; then
			./scripts/prepare_files.sh extensor_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.mtx 	
		elif [ $2 -eq 0 ]; then
			./scripts/prepare_files_no_gold.sh extensor_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.mtx
		fi
		for b in ${!BENCHMARKS[@]}; do
			bench=${BENCHMARKS[$b]}
			
			export TILED_SUITESPARSE_FORMATTED_PATH=${SAM_HOME}/tiles/$bench/formatted
			export TILED_OUTPUT_PATH=${SAM_HOME}/tiles/$bench/output/
			mkdir -p $path
			echo "Testing $bench..."

			line=random_sparsity
			cd $basedir/sam/sim
			if [ $2 -eq 1 ]; then
				pytest test/advanced-simulator/test_$bench"_tile_pipeline_final".py --ssname $line -s --check-gold --skip-empty --nbuffer --yaml_name=$1 --nnz-value=${NNZ[$nnz]} --benchmark-json=$path/${line}_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.json 
			else
				pytest test/advanced-simulator/test_$bench"_tile_pipeline_final".py --ssname $line -s --skip-empty --nbuffer --yaml_name=$1 --nnz-value=${NNZ[$nnz]} --benchmark-json=$path/${line}_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.json 
			fi
			python $basedir/scripts/converter.py --json_name $path/${line}_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.json	
			    
			status=$?
			if [ $status -gt 0 ]
			then 
			  errors+=("${line}, ${bench}")
			fi
			cd $basedir
		done
	done
	
	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done


for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	bench_name=$bench"_tile_pipeline_final"
	python3 $basedir/scripts/bench_csv_aggregator.py $path $basedir/$benchout/$bench_name.csv
done
popd

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
printf "$ELAPSED"
printf "\n"

