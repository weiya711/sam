#reading one file in this folder, taking matrix name, and median time for elemadd

#**Note: main idea, use this base script to extract all the median times and the name
import os
def read_file_and_extract_data(input_file, output_file, target_string):
    print(input_file)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    found = False
    data = []
    for line in lines:
        if target_string in line:
            values = line.strip().split(',')
            print(values)
            if len(values) >= 3:
                #run script with a 7 here for all the matrix names in order, and 3 for all the cpu_times in order, 12 for the nnz(number of non zeros) in order
                data.append(values[3])
            with open(output_file, 'a') as file:
                file.write('\n')
                file.write('\n'.join(data))
                print("Data extracted and saved to", output_file)


#calling function:
#*Note: can change file name so saved for small/medium/large and titled accordingly
#can change output file path for each 'application' running - ex below is set up for elemadd
#output_file_path = 'result_large_compilation_file_elemadd.txt'
#target_string = 'bench_suitesparse/mat_elemadd_mmadd/iterations:1/real_time_median'

#set up below is for vecmul
#output_file_path = 'result_large_compilation_file_vecmul_spmv.txt'
#target_string = 'bench_suitesparse/vecmul_spmv/iterations:1/real_time_median'

#set up below is for elemadd3
#output_file_path = 'result_large_compilation_file_elemadd3_spmv.txt'
#target_string = 'bench_suitesparse/mat_elemadd3_plus3/iterations:1/real_time_median'

#set up below is for sddmm
#output_file_path = 'result_large_compilation_file_sddmm.txt'
#target_string = 'bench_suitesparse/mat_sddmm/iterations:1/real_time_median'

#set up below is for residual
#output_file_path = 'result_compilation_file_residual.txt'
#target_string = 'bench_suitesparse/mat_residual/iterations:1/real_time_median'

#set up below is for mattransmul
#output_file_path = 'result_med_compilation_file_mattransmul.txt'
#target_string = 'bench_suitesparse/mat_mattransmul/iterations:1/real_time_median'

#set up below is for spmm
#output_file_path = 'result_med_compilation_file_matmul_spmm.txt'
#target_string = 'bench_suitesparse/matmul_spmm/iterations:1/real_time_median'

#set up below is for elemmul
output_file_path = 'result_med_compilation_file_elemmul.txt'
target_string = 'bench_suitesparse/mat_elemmul/iterations:1/real_time_median'

def main():
    for dirName, subDirList, fileList in os.walk('/home/jadivara/sam/suitesparse-bench/taco'):
        for file in fileList:
            input_file_path = file
            read_file_and_extract_data(input_file_path, output_file_path, target_string)
main()