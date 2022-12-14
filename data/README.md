# Suitesparse Data

Example formatted data for running SAM simulations and Onyx HW

We have committed two example Suitesparse matrices for inspection (both in .mtx format and formatted for SAM). 
All mtx files can be found in `suitesparse/` and all files formatted for SAM can be found in `suitesparse-formatted`

To run SAM on all other suitesparse, follow these steps:
  1. Start from the `sam/` rootdir
  2. Download all other suitesparse matrices using 
  ```
  ./scripts/download_suitesparse.sh
  ```
  3. Reformat to create expression and SAM specific files using: 
  ```
  ./scripts/generate_suitesparse_formats.sh <tensor_name_txt>
  ```

# Frostt Data

TODO
