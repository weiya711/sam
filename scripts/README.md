# Tags
1. `cpu-taco` means it is used to get baseline runtime for TACO running on the CPU
2. `sam` means it is used to get cycles for SAM
3. `format` means it is used to format the data into the correct datastructures for SAM
4. `ss` means it is used to run SuiteSparse benchmarks
5.  `frostt` means it is used to run FROSTT benchmarks
6.  `synth` means it is used to run synthetic data benchmarks
7.  `machine` means it needs an argument to tell the script about the machine running (local, Neva/Kiwi, or Lanka)
8.  `sam-mem` means it is used for the SAM memory modeling simulator
9.  `artifact` means it is used for the [SAM ASPLOS '23 artifact evaluation](https://github.com/weiya711/sam-artifact) (ASPLOS23 AE)
10.  `plot` means it is used to plot data

# Command and Desciption

1. `./scriptsadvanced_simulator_runner.sh ... TODO`
 
2. `python scripts/artifact_docker_copy.py --output_dir <OUT_DIR> --docker_id <CONTAINER_ID>`
    
    Tags: `artifact` 
    Description: Extracts all figures from docker to your local machine (see [Validate All Results](https://github.com/weiya711/sam-artifact#Validate-All-Results) in the SAM artifact evaluation README
3. `python scripts/bench_csv_aggregator.py <IN_DIR> <OUT_CSV> [--taco] [--label_set_file <FILE>]`

    Tags: `cpu-taco`, `sam`, `ss`, `frostt`  
    Description: Aggregates all csvs from a directory into one file. Can do this for either the TACO-generated (`--taco`) or SAM-generated CSVs
4. `./scripts/clean_memory_model.sh` 

    Tags: `sam-mem`, `artifact`
    Description: Cleans all directories related to the SAM memory modeling simulator and is used in the ASPLOS23 AE
5. `python scripts/collect_node_counts.py [--sam_graphs <SAM_DIR> --output_log <OUT_LOG>]

    Tags: `artifact`
    Description: `make sam` must be run before this script. This generates Table 1 in the ASPLOS '23 SAM paper for the AE
6. `python converter.py ... TODO`
    Tags: `sam`
    Description: Converts JSONs to CSVs for SAM/pytest benchmarks
7. `python scripts/datastructure_suitesparse.py ... TODO` 
    Note: Do not use this, the `generate_suitesparse_formats.sh` script should be used instead
8. `python datastructure_tns.py` 
    TODO
9. `python divvy_runs.py`
    TODO


43. `./scripts/suitesparse_runner.sh <TXT> <0|1|2>`

    Tags: `cpu-taco`, `ss`, `machine`   
    Description: Gets the TACO CPU runtime baselines for SuiteSparse and stores it to `suitesparse-bench/taco/`

50. 
