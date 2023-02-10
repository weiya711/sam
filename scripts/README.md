| File Name | Usage | Description |
-----------------------------------
| `./scripts/suitesparse_runner.sh <TXT> <0|1|2>` | `cpu-taco`, `ss`, `machine`   | Gets the TACO CPU runtime baselines for SuiteSparse and stores it to `suitesparse-bench/taco/` |
| `python scripts/bench_csv_aggregator.py <IN_DIR> <OUT_CSV> [--taco] [--label_set_file <FILE>]` | `cpu-taco`, `sam`, `ss`, `frostt`  | Aggregates all csvs from a directory into one file. Can do this for either the TACO-generated (`--taco`) or SAM-generated CSVs |
 
