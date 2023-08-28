#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/prepare_files.sh extensor_<NNZ>_<DIM>.mtx

basedir=$(pwd)

rm -rf $basedir/tiles/*

./scripts/tiling/tile_ext.sh $1 memory_config_extensor_17M_llb.yaml

python scripts/tiling/generate_gold_matmul_tiled.py --yaml_name memory_config_extensor_17M_llb.yaml
