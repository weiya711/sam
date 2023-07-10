#reading one file in this folder, taking matrix name, and median time for elemadd

#**Note: main idea, use this base script to extract all the median times and the name
import os
def read_file_and_extract_data(input_file, output_file):
    print(input_file)
    nnz = []
    x = 0
    with open(input_file, 'r') as file:
        x = len(file.readlines())
        print(x)
    with open(output_file, 'a') as file:
        file.write('\n')
        file.write(''.join(input_file))
        #nnz.append(x)
        file.write(''.join(str(x)))
        print("Data extracted and saved to", output_file)

#calling function:
#*Note: can change file name so saved for small/medium/large and titled accordingly
#can change output file path for each 'application' running - ex below is set up for ttv

def main():
    output_file_path = 'nnznums.txt'
    file_path = os.environ['FROSTT_PATH']
    for dirName, subDirList, fileList in os.walk(file_path):
        for file in fileList:
            input_file_path = file
            read_file_and_extract_data(input_file_path, output_file_path)
main()