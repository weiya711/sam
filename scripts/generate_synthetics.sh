#!/bin/bash

# Vars
if [ -z "$2" ]
then
    export SYNTHETIC_PATH="$(pwd)/synthetic/"
else
    export SYNTHETIC_PATH="$2"
fi

export SRC_PATH="$(pwd)/sam/onyx/synthetic/"

# Create the main directories
pushd $SYNTHETIC_PATH
for vectype in "random" "blocks" "runs"
do
    mkdir -p "${SYNTHETIC_PATH}/${vectype}/compressed/"
    mkdir -p "${SYNTHETIC_PATH}/${vectype}/uncompressed/"
    case $vectype in
        random)
            python ${SRC_PATH}/generate_random_mats.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --name B --shape 2000 --output_format CSF
            python ${SRC_PATH}/generate_random_mats.py --seed 1 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --name C --shape 2000 --output_format CSF
            python ${SRC_PATH}/generate_random_mats.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --name B --shape 2000 --output_format UNC
            python ${SRC_PATH}/generate_random_mats.py --seed 1 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --name C --shape 2000 --output_format UNC
            ;;
        blocks)
            python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --shape 2000 --output_format CSF
            python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --shape 2000 --output_format UNC
            ;;
        runs)
            python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --output_format CSF --run_lengths 100 200
            python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --output_format UNC --run_lengths 100 200
            ;;
    esac
done
popd

# Now generate the matrices in both DCSR/DCSC formats
pushd $SYNTHETIC_PATH

mkdir -p "${SYNTHETIC_PATH}/matrix/DCSR"
mkdir -p "${SYNTHETIC_PATH}/matrix/DCSC"
mkdir -p "${SYNTHETIC_PATH}/matrix/DENSE"

i=50
j=50
k=20

sparsity="0.95"

python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name B --shape $i $k --output_format CSF
python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSC/ --name B --shape $i $k --output_format CSF --transpose
python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DENSE/ --name B --shape $i $k --output_format UNC

python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name C --shape $k $j --output_format CSF
python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSC/ --name C --shape $k $j --output_format CSF --transpose
python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DENSE/ --name C --shape $k $j --output_format UNC

popd