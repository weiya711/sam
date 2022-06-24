#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH --exclusive

SAMNAME=(
  matmul_ikj
  vecmul_ij
  #mat_elemadd
  #mat_residual
  #mat_sddmm
  #mat_elemadd3
  #mat_mattransmul
)

TACONAME=(
  matmul_spmm
  vecmul_spmv
  #mat_elemadd_mmadd
  #mat_residual
  #mat_sddmm
  #mat_elemadd3_plus3
  #mat_mattransmul
)

set -u

sspath=/nobackup/owhsu/sparse-datasets/suitesparse
vout=/nobackup/owhsu/validate

mkdir -p "$vout"

while read line; do
	matrix="$sspath/$line.mtx"
    
    # TACO
    GEN=ON SUITESPARSE_TENSOR_PATH="$matrix" make -j8 validate-bench BENCHES="bench_suitesparse" VALIDATION_OUTPUT_PATH="$vout" NEVA=ON 

    cd sam/sim
    # SAM
    for b in ${!SAMNAME[@]}; do
        sambench=${SAMNAME[$b]}
        taconame=${TACONAME[$b]}
        sam_vout=${vout}/$line-$taconame-sam.tns
        pytest test/apps/test_$sambench.py --ssname $line --result-out ${sam_vout} 
    done
        
done <$1



