dir=./sam-outputs
dot_dir=$dir/dot
png_dir=$dir/png
taco_exe=./taco/build/bin/taco


mkdir -p $dir
mkdir -p $dot_dir
mkdir -p $png_dir


name=matmul_kij
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss -f=B:ss:1,0 -f=C:ds --print-sam-graph="$dot_dir/$name.gv" -s="reorder(k,i,j)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png


name=matmul_kji
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ds --print-sam-graph="$dot_dir/$name.gv" -s="reorder(k,j,i)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=matmul_ikj
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss -f=B:ss -f=C:ds --print-sam-graph="$dot_dir/$name.gv" -s="reorder(i,k,j)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=matmul_jki
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss:1,0 -f=B:ss -f=C:ds:1,0 --print-sam-graph="$dot_dir/$name.gv" -s="reorder(j,k,i)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=matmul_ijk
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss -f=B:ss -f=C:ds:1,0 --print-sam-graph="$dot_dir/$name.gv" -s="reorder(i,j,k)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=matmul_jik
$taco_exe "X(i,j)=B(i,k)*C(k,j)" -f=X:ss:1,0 -f=B:ss -f=C:ds:1,0 --print-sam-graph="$dot_dir/$name.gv" -s="reorder(j,i,k)" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=mat_elemmul
$taco_exe "X(i,j)=B(i,j)*C(i,j)" -f=X:ss -f=B:ss -f=C:ds --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=vecmul
$taco_exe "x(i)=B(i,j)*c(j)" -f=x:s -f=B:ds -f=c:s --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=vec_elemmul
$taco_exe "x(i)=b(i)*c(i)" -f=x:s -f=b:s -f=c:d --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=vec_elemadd
$taco_exe "x(i)=b(i)+c(i)" -f=x:s -f=b:s -f=c:d --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=vec_scalar_mul
$taco_exe "x(i)=b*c(i)" -f=x:s -f=c:s --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png

name=tensor_elemmul
$taco_exe "X(i,j,k)=B(i,j,k)*C(i,j,k)" -f=X:dss -f=B:sss -f=C:sds --print-sam-graph="$dot_dir/$name.gv" --print-concrete
dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png


