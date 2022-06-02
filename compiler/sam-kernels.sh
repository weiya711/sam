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
  mat_elemadd
  mat_identity
  vecmul_ij
  vecmul_ji
  vec_elemmul
  vec_identity
  vec_elemadd
  vec_scalar_mul
  tensor3_elemmul
  tensor3_identity
  tensor3_elemadd
  tensor3_innerprod
  tensor3_ttv
  tensor3_ttm
  mat_sddmm
  mat_mattransmul
  mat_residual
  mat_elemadd3
  tensor3_mttkrp
)


TACO_ARGS=(
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss:1,0 -f=C:ss -s=reorder(k,i,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss -s=reorder(k,j,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss -s=reorder(i,k,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss:1,0 -s=reorder(j,k,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss -f=C:ss:1,0  -s=reorder(j,i,k)"
  "X(i,j)=B(i,j)*C(i,j) -f=X:ss -f=B:ss -f=C:ss"
  "X(i,j)=B(i,j)+C(i,j) -f=X:ss -f=B:ss -f=C:ss"
  "X(i,j)=B(i,j) -f=X:ss -f=B:ss"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss -f=c:s"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss:1,0 -f=c:s -s=reorder(j,i)"
  "x(i)=b(i)*c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b(i) -f=x:s -f=b:s"
  "x(i)=b(i)+c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b*c(i) -f=x:s -f=c:s"
  "X(i,j,k)=B(i,j,k)*C(i,j,k) -f=X:sss -f=B:sss -f=C:sss"
  "X(i,j,k)=B(i,j,k) -f=X:sss -f=B:sss"
  "X(i,j,k)=B(i,j,k)+C(i,j,k) -f=X:sss -f=B:sss -f=C:sss"
  "x=B(i,j,k)*C(i,j,k) -f=B:sss -f=C:sss"
  "X(i,j)=B(i,j,k)*c(k) -f=X:ss -f=B:sss -f=c:s"
  "X(i,j,k)=B(i,j,l)*C(k,l) -f=X:sss -f=B:sss -f=C:ss"
  "X(i,j)=B(i,j)*C(i,k)*D(k,j) -f=X:ss -f=B:ss -f=C:dd -f=D:dd:1,0 -s=reorder(i,j,k)"
  "x(i)=b*C(j,i)*d(j)+e*f(i) -f=C:ds:1,0"
  "x(i)=b(i)-C(i,j)*d(j) -f=C:ds"
  "X(i,j)=B(i,j)+C(i,j)+D(i,j) -f=X:ss -f=B:ss -f=C:ss -f=D:ss"
  "X(i,j)=B(i,k,l)*C(j,k)*D(j,l) -f=X:ss -f=B:sss -f=C:ss -f=D:ss" 
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

