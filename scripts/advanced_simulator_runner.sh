#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

BENCHMARKS=(
##  mat_vecmul_FINAL
#  matmul_FINAL
#  mat_identity
#  mat_identity_back
#  matmul_ikj_memory_back
   matmul_ikj_lp
#  matmul_ikj_tiled_bcsstm02
#  matmul_ikj_check
#  matmul_ikj_tiling
#  matmul_ikj_back
#  mat_elemmul_FINAL
#  mat_elemadd_FINAL
#  mat_elemadd3_FINAL
#  mat_residual_FINAL
#  mat_mattransmul_FINAL
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

# LANKA
if [ $2 -eq 1 ]; then
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
elif [ $2 -eq 2 ]; then
	export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
	export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
	export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
	export FROSTT_FORMATTED_TACO_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted/taco-tensor
	export FROSTT_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted
	export TACO_TENSOR_PATH=/nobackup/owhsu/sparse-datasets
	export TILED_SUITESPARSE_FORMATTED_PATH=/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/formatted
	export TILED_OUTPUT_PATH=/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/output/
	lanka=OFF
	neva=ON
else
	lanka=OFF
	neva=OFF
fi

format_outdir=${SUITESPARSE_FORMATTED_PATH} 
basedir=$(pwd)
sspath=$SUITESPARSE_PATH
benchout=suitesparse-bench_simulator/sam

source $basedir/../venv/bin/activate

#__conda_setup="$('/data/scratch/owhsu/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/data/scratch/owhsu/miniconda/etc/profile.d/conda.sh" ]; then
#        . "/data/scratch/owhsu/miniconda/etc/profile.d/conda.sh"
#    else
#        export PATH="/data/scratch/owhsu/miniconda/bin:$PATH"
#    fi
#fi
#unset __conda_setup
#conda activate aha

mkdir -p "$benchout"
mkdir -p $format_outdir
mkdir -p $TACO_TENSOR_PATH/other-formatted-taco

make -j8 taco/build NEVA=$neva LANKA=$lanka GEN=ON

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	while read line; do
		cd $format_outdir

		if [ $2 -eq 1 ]; then
			matrix="$sspath/$line/$line.mtx"
		elif [ $2 -eq 2 ]; then
			matrix="$sspath/$line.mtx"
		else
			matrix="$sspath/$line.mtx"
		fi

		if [ "$bench" == "mat_vecmul_FINAL" ]; then
			echo "Generating input format files for $line..."
			SUITESPARSE_TENSOR_PATH=$matrix python $basedir/scripts/datastructure_suitesparse.py -n $line 

			SUITESPARSE_TENSOR_PATH=$matrix $basedir/compiler/taco/build/bin/taco-test sam.pack_other_ss    
			python $basedir/scripts/datastructure_frostt.py -n $line -f ss01 --other -ss
		fi

		cd $basedir/sam/sim
		#python -m cProfile -o test/final-apps/test_$bench.py --ssname $line -s --benchmark-json=$path/$line.json 
		pytest test/advanced-simulator/test_$bench.py --ssname $line -s  --report-stats --check-gold --benchmark-json=$path/$line.json 
		python $basedir/scripts/converter.py --json_name $path/$line.json	
		    
		status=$?
		if [ $status -gt 0 ]
		then 
		  errors+=("${line}, ${bench}")
		fi

		cd $basedir
	done <$1

	python3 $basedir/scripts/bench_csv_aggregator.py $path $basedir/$benchout/suitesparse_$bench.csv

	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done
