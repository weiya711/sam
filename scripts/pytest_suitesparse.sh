#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360
outdir=/nobackup/owhsu/sparse-datasets/suitesparse-formatted

DATASET_NAMES=(
#  bcsstm04
  bcsstm02
  lpi_itest2
#  bcsstm03
#  lpi_bgprtr
#  cage4
#  klein-b1
#  GD02_a
#  GD95_b
#  Hamrle1
#  LF10
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


mkdir -p $outdir
cd ./sam/sim

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 

    echo "Testing $name..."

    pytest -k test_matmul_ --ssname $name -s  #--debug-sim 
    status=$?
    if [ $status -gt 0 ]
    then 
      errors+=("${name} matmul_ijk")
    fi
 

#
#    pytest -k test_matmul_ijk_i --ssname $name -s #--debug-sim 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk")
#    fi
   
#    pytest -k test_mat_elemmul_i --ssname $name -s 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk")
#    fi
 

#    pytest -k test_tensor3_elemmul_i --ssname $name -s 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk")
#    fi

#    pytest -k test_matmul_jik_i --ssname $name -s 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk")
#    fi

#    pytest -k test_matmul_jki_i --ssname $name -s 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk")
#    fi




#    pytest -k test_mat_identity_i --ssname $name -s
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} mat_identity")
#    fi

#    pytest -k test_mat_elemmul_i --ssname $name -s
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      	    errors+=("${name} mat_identity")
#    fi



done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
