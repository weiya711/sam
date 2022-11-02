
rm -rf $basedir/tiles/*

./scripts/tile_ext.sh $1 memory_config_extensor_17M_llb.yaml

python scripts/generate_gold_matmul_tiled.py

./scripts/advanced_simulator_runner.sh temp.txt 3 memory_config_extensor_17M_llb.yaml
