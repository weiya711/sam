#!/bin/bash
#set -xe
#SBATCH -N 1
#SBATCH -t 360

dir=./sam-outputs
dot_dir=$dir/dot
png_dir=$dir/png
taco_exe=./taco/build/bin/taco

KERNEL_NAMES=(
  tensor3_linear
  tensor4_multiply2
)


TACO_ARGS=(
  "X(i,j,k)=B(j,l)*C(i,l,k)+D(j) -f=X:sss:0,2,1 -f=B:ss:1,0 -f=C:sss:0,2,1 -s=reorder(i,k,l,j)" #reorder: jikl 
  "X(i,j,k)=B(j,l)*C(i,l,k) -f=X:sss:0,2,1 -f=B:ss:1,0 -f=C:sss:0,2,1" #reorder: jikl 
  "X(i,j,k)=C(i,l,k)+D(j) -f=X:sss:0,2,1 -f=B:ss:1,0 -f=C:sss:0,2,1 -s=reorder(i,k,l,j)" #reorder: jikl 
  # "X(i,k,j,m)=B(i,j,k,l)*C(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=C:ssss:0,3,1,2 -s=reorder(i,k,j,m,l)"
  # "X(i,k,j,m)=B(i,j,k,l)*C(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=C:ssss:0,3,1,2 -s=reorder(i,k,j,m,l)"
  "X(i,k,j,m)=B(i,j,k,l)*C(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=C:ssss:0,2,3,1 -s=reorder(i,k,j,m,l)"
)

mkdir -p $dir
mkdir -p $dot_dir
mkdir -p $png_dir

for i in ${!KERNEL_NAMES[@]}; do
    name=${KERNEL_NAMES[$i]}
    args=${TACO_ARGS[$i]}

    $taco_exe $args --print-sam-graph="$dot_dir/$name.gv"
    dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png
    echo "Generating sam for $name to $dir"
done

