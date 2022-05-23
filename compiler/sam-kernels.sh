#!/bin/bash
#set -xe
#SBATCH -N 1
#SBATCH -t 360

dir=./sam-outputs
dot_dir=$dir/dot
png_dir=$dir/png
taco_exe=./taco/build/bin/taco

KERNEL_NAMES=(
  matmul_kij
  matmul_kji
  matmul_ikj
  matmul_jki
  matmul_ijk
  matmul_jik
  mat_elemmul
  mat_identity
  vecmul_ij
  vecmul_ji
  vec_elemmul
  vec_identity
  vec_elemadd
  vec_scalar_mul
  tensor3_elemmul
  tensor3_identity
)

TACO_ARGS=(
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss:1,0 -f=C:ss -s=reorder(k,i,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss -s=reorder(k,j,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss -s=reorder(i,k,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss:1,0 -s=reorder(j,k,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss -f=C:ss:1,0  -s=reorder(j,i,k)"
  "X(i,j)=B(i,j)*C(i,j) -f=X:ss -f=B:ss -f=C:ds"
  "X(i,j)=B(i,j) -f=X:ss -f=B:ss"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss -f=c:s"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss:1,0 -f=c:s -s=reorder(j,i)"
  "x(i)=b(i)*c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b(i) -f=x:s -f=b:s"
  "x(i)=b(i)+c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b*c(i) -f=x:s -f=c:s"
  "X(i,j,k)=B(i,j,k)*C(i,j,k) -f=X:sss -f=B:sss -f=C:sss"
  "X(i,j,k)=B(i,j,k) -f=X:sss -f=B:sss"
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

