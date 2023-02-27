BENCHMARKS=(
tiling
unit_reorder
)

# Vars
export SYNTHETIC_PATH="$(pwd)/synthetic/"

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

format_outdir=${SUITESPARSE_FORMATTED_PATH} 
basedir=$(pwd)
sspath=$SUITESPARSE_PATH
benchout=suitesparse-bench/sam

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

mkdir -p "$benchout"
mkdir -p $format_outdir

make -j8 taco/build NEVA=$neva LANKA=$lanka GEN=ON



mkdir -p "${SYNTHETIC_PATH}/matrix/DCSR"
mkdir -p "${SYNTHETIC_PATH}/matrix/DCSC"
mkdir -p "${SYNTHETIC_PATH}/matrix/DENSE"



for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	while read line; do
		cd $format_outdir

		matrix="$sspath/$line.mtx"

		cd $basedir/sam/sim

		pytest test/advanced-simulator/test_$bench.py --ssname $line -s --report-stats --split-factor=$2 --benchmark-json=$path/$line.json 
		python $basedir/scripts/converter.py --json_name $path/$line.json	
		    		    
		status=$?
		if [ $status -gt 0 ]
		then 
		  errors+=("${line}, ${bench}")
		fi

		cd $basedir	
	done <$1

	python $basedir/scripts/bench_csv_aggregator.py $path $basedir/$benchout/suitesparse_$bench.csv

	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done
