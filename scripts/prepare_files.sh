#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive
#SBATCH --mail-user=oliviahsu1107@gmail.com

basedir=$(pwd)

rm -rf $basedir/tiles/*

./scripts/tile_ext.sh $1 memory_config_extensor_17M_llb.yaml

python scripts/generate_gold_matmul_tiled.py --yaml_name memory_config_extensor_17M_llb.yaml
