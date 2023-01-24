#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive
SECONDS=0
set -u

BENCHMARKS=(
	matmul_ikj_tile_pipeline_final
)


NNZ=(
  5000
#  10000
#  25000
#  50000
)

DIMENSIONS=(
 	#1024
# 2360
 3696
 6368
 9040
 11712
 13716
 15720
 
 # 5032
# 7704
# 9040
# 11712
# 15720
)


errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

basedir=$(pwd)

export SAM_HOME=$basedir
export TILED_SUITESPARSE_FORMATTED_PATH=${SAM_HOME}/tiles/matmul_ikj/formatted
export TILED_OUTPUT_PATH=${SAM_HOME}/tiles/matmul_ikj/output/

benchout=suitesparse-bench_simulator/sam

mkdir -p "$benchout"

for b in ${!BENCHMARKS[@]}; do
	for nnz in ${!NNZ[@]}; do
		for dim in ${!DIMENSIONS[@]}; do
			if [ $2 -eq 0 ]; then
				./scripts/prepare_files_no_gold.sh extensor_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.mtx
			elif [ $2 -eq 1 ]; then
				./scripts/prepare_files.sh extensor_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.mtx
			fi
			bench=${BENCHMARKS[$b]}
			path=$basedir/$benchout/$bench
			mkdir -p $basedir/$benchout/$bench
			echo "Testing $bench..."

			line=random_sparsity
			cd $basedir/sam/sim
			if [ $2 -eq 1 ]; then
				pytest test/advanced-simulator/test_$bench.py --ssname $line -s --check-gold --skip-empty --nbuffer --yaml_name=$1 --nnz-value=${NNZ[$nnz]} --benchmark-json=$path/${line}_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.json 
			else
				pytest test/advanced-simulator/test_$bench.py --ssname $line -s --skip-empty --nbuffer --yaml_name=$1 --nnz-value=${NNZ[$nnz]} --benchmark-json=$path/${line}_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.json 
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
	python3 $basedir/scripts/bench_csv_aggregator.py $path $basedir/$benchout/suitesparse_$bench.csv
	
	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
printf "$ELAPSED"
