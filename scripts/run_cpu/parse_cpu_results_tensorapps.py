#reading one file in this folder, taking matrix name, and median time for elemadd

#**Note: main idea, use this base script to extract all the median times and the name

#place in the results-cpu folder where cpu results for tensor apps populate, and then can run this script using the following command:
#python parse_cpu_results_tensorapps.py

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
                data.append(values[7])
            with open(output_file, 'a') as file:
                file.write('\n')
                file.write('\n'.join(data))
                print("Data extracted and saved to", output_file)


#calling function:
#*Note: can change file name so saved for small/medium/large and titled accordingly
#can change output file path for each 'application' running - ex below is set up for ttv
# output_file_path = 'result_tensor_compilation_file_ttv.txt'
# target_string = 'tensor3_ttv/iterations:1/real_time_median'

#set up below is for innerprod
#output_file_path = 'result_tensor_compilation_file_innerprod.txt'
#target_string = 'tensor3_innerprod/iterations:1/real_time_median'

#set up below is for elemadd_plus2
output_file_path = 'result_tensor_compilation_file_elemadd_plus2.txt'
target_string = 'tensor3_elemadd_plus2/iterations:1/real_time_median'

#set up below is for ttm
#output_file_path = 'result_tensor_compilation_file_ttm.txt'
#target_string = 'tensor3_ttm/iterations:1/real_time_median'

#set up below is for mttkrp
# output_file_path = 'result_tensor_compilation_file_mttkrp.txt'
# target_string = 'tensor3_mttkrp/iterations:1/real_time_median'

def main():
    for dirName, subDirList, fileList in os.walk('/nobackup/jadivara/sam/results-cpu'):
        for file in fileList:
            input_file_path = file
            read_file_and_extract_data(input_file_path, output_file_path, target_string)
main()