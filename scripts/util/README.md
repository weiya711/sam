# Utilities

The `scripts/util` folder contains util.py (shared utility functions) and
shared utility scripts used to aggregate and format csv data.

1. `util.py` -  
2. `bench_csv_aggregator.py` - Script that aggregates all of the output CSVs.
   This is useful since CPU tests are run using googlebench potentially one
   tensor at a time (to run tests in parallel), which will produce one CSV per tensor. 
3. `converter.py` - Converts JSON to CSV. 
