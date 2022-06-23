#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360
outdir=/nobackup/owhsu/sparse-datasets/frostt-formatted

DATASET_NAMES=(
   facebook
   fb10k
   fb1k
   nell-1
   nell-2
   taco-tensor
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt
export FROSTT_FORMATTED_PATH=$outdir

mkdir -p $outdir
cd ./sam/sim

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 

    echo "Testing $name..."
#    pytest -k test_mat_mul_ijk_csr_full_i --ssname $name 
#    status=$?
#    if [ $status -gt 0 ]
#    then 
#      errors+=("${name} matmul_ijk_full")
#    fi

#    pytest -k test_mat_identity_i --ssname $name -s 
#    status=$?
#    if [ $status -gt 0 ]
#    then
#      errors+=("${name} matmul_ijk")
#    fi
 
    pytest -k test_tensor --frosttname $name -s -vv #--debug-sim 
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
