#!/bin/bash
#set -xe
#SBATCH -N 1
#SBATCH -t 360

dir=./sam-outputs
dot_dir=$dir/dot
png_dir=$dir/png
taco_exe=./taco/build/bin/taco

KERNEL_NAMES=(
#   tensor3_linear
  # tensor3_fused_feedforward_linear
  # test_max
  # tensor4_mult2_ijklm
  tensor4_mult2_ijkml
  tensor4_mult2_ikjml
  # tensor3_softmax_multiply2
  # tensor3_fusedlinear1
  # tensor4_mult
  # tensor4_fused
  # tensor3_fused_ffn
  # tensor4_multiply_ijklm
  # tensor4_multiply2_ijklm
  # tensor4_fused_ijklm
)


TACO_ARGS=(
  # "X(i,j,k)=B(i,j,k)*S(i,j) -f=X:sss:0,1,2 -f=M:ss:0,1 -f=D:sss:0,1,2" #reorder: jikl 
  # "X(i,m,k)=(E(m,j)*B(j,l)*C(i,l,k)+E(m,j)*d(j))+f(m)" #-f=X:sss:1,2,0 -f=B:ss:0,1 -f=C:sss:2,0,1 -f=d:s -s=reorder(j,k,i)" #reorder: jikl 
#   "X(i,m,k)=E(m,j)*(B(j,l)*C(i,l,k)+d(j))+f(m)" #-f=X:sss:1,2,0 -f=B:ss:0,1 -f=C:sss:2,0,1 -f=d:s -s=reorder(j,k,i)" #reorder: jikl
#  "X(i,j,k)=exp(B(i,j,k))"
  # "X(i,j,k)=B(j,l)*C(i,l,k)+d(j) -f=X:sss:1,0,2 -f=B:ss:0,1 -f=C:sss:0,1,2 -f=d:s -s=reorder(j,i,k)" #reorder: jikl 
  # "X(i,j,k,l)=Q(i,k,j,m)*K(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=C:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
#   "X(i,j,k)=B(j,l)*C(i,l,k) -f=X:sss:0,2,1 -f=B:ss:1,0 -f=C:sss:0,2,1" #reorder: jikl 
#   "X(i,j,k)=C(i,l,k)+D(j) -f=X:sss:0,2,1 -f=B:ss:1,0 -f=C:sss:0,2,1 -s=reorder(i,k,l,j)" #reorder: jikl 
  # "X(i,k,j,m)=B(i,j,k,l)*C(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=C:ssss:0,3,1,2 -s=reorder(i,k,j,m,l)"
  # "X(i,k,j,m)=B(i,j,k,l)*C(i,l,j,m) -f=X:ssss:0,2,1,3 -f=B:ssss:0,1,2,3 -f=C:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
  # "X(i,k,j,m)=Q(i,k,j,m)*K(i,l,j,m)*V(i,l,j,m) -f=X:ssss:0,2,1,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,2,1,3 -f=V:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
  # "X(i,j,k,l)=Q(i,k,j,m)*K(i,l,j,m) -f=X:ssss:0,1,2,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
  # "X(i,k,j,m)=B(i,j,k,l)*V(i,l,j,m) -f=X:ssss:0,2,1,3 -f=B:ssss:0,1,2,3 -f=V:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
  "X(i,k,j,m)=B(i,j,k,l)*V(i,l,j,m) -f=X:ssss:0,2,1,3 -f=B:ssss:0,1,2,3 -f=V:ssss:0,2,3,1 -s=reorder(i,j,k,m,l)"
  "X(i,k,j,m)=B(i,j,k,l)*V(i,l,j,m) -f=X:ssss:0,1,2,3 -f=B:ssss:0,2,1,3 -f=V:ssss:0,2,3,1 -s=reorder(i,k,j,m,l)"
  # "X(i,k,j,m)=(Q(i,k,j,m)*K(i,l,j,m))*V(i,l,j,m) -f=X:ssss:0,2,1,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,2,1,3 -f=V:ssss:0,2,1,3 -s=reorder(i,j,k,l,m)"
   # "X(i,k,j,m)=(Q(i,k,j,m)*K(i,l,j,m))*V(i,l,j,m) -f=X:ssss:0,2,1,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,2,1,3 -f=V:ssss:0,2,1,3 -s=reorder(i,j,k,m,l)"
#   "X(i,k,j,m)=Q(i,k,j,m)*K(i,l,j,m) -f=X:ssss:0,1,2,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,3,1,2 -s=reorder(i,k,j,m,l)"
  # "X(i,k,j,n)=(Q(i,k,j,m)*K(i,l,j,m))*V(i,l,j,n) -f=X:ssss:0,2,1,3 -f=Q:ssss:0,2,1,3 -f=K:ssss:0,2,1,3 -f=V:ssss:0,2,1,3 -s=reorder(i,j,k,l,m,n)"
  # "X(i,k,j,m)=(Q(i,k,j,m)*K(i,l,j,m))*V(i,l,j,m)"
  # "X(i,j)=B(i,j)+C(j) -f=B:ss -f=C:s"
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

