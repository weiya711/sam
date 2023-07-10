#script to generate 50 random 3D tensors (seeded, produces same 50 each time)
import numpy as np
import random
import os

# Set the seed value
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)


#generating tensor dimensions and storing results in an array (array size is 150, 50 tensors and 3 dimensions for each tensor)
#conditions which need to be met for each set of 3 tensor dimensions: no dimension can't be 0, and can't have a tensor with more than 900 elements (meaning dimension1*dimension2*dimension3 <= 900)
dimensions = [0] * 150
dimensions_onetensor = [0] * 3
# x goes from 0 to 49
for x in range(50):
    dimensions_onetensor[0] = random.randint(1,60)
    dimensions_onetensor[1] = random.randint(1,60)
    dimensions_onetensor[2] = random.randint(1,60)

    while((dimensions_onetensor[0]*dimensions_onetensor[1]*dimensions_onetensor[2])>901):
        dimensions_onetensor[0] = random.randint(1,60)
        dimensions_onetensor[1] = random.randint(1,60)
        dimensions_onetensor[2] = random.randint(1,60)

    #print(dimensions_onetensor[0])
    #print(dimensions_onetensor[1])
    #print(dimensions_onetensor[2])
    dimensions[x*3] = dimensions_onetensor[0]
    dimensions[(x*3)+1] = dimensions_onetensor[1]
    dimensions[(x*3)+2] = dimensions_onetensor[2]
    #print(dimensions[x*3])
    #print(dimensions[x*3+1])
    #print(dimensions[x*3+2])

    dimensions_onetensor[0] = 0
    dimensions_onetensor[1] = 0
    dimensions_onetensor[2] = 0
    #print('\n')

# y goes from 0 to 149
#for y in range(150):
#    print(dimensions[y])
#    if((y+1)%3==0):
#        print('\n')


#Generating tensor values based on the dimensions now stored in the dimensions (150 elem) array
#i goes from 0 to 49
tensor_num = 1
randomNumber = 0
numToInsert = 0
frostt_path = os.environ['FROSTT_PATH']
for i in range(50):
    filename = os.path.join(frostt_path, "tensor"+str(tensor_num)+".tns")
    f = open(filename, "a")
    lineToAddInFile = ""
    #f.write('dimensions:' + '\n')
    #f.write(""+str(dimensions[i*3]) + " " + str(dimensions[i*3+1]) + " " + str(dimensions[i*3+2]))
    #f.write("\n")
    arr = np.zeros([dimensions[i*3],dimensions[(i*3)+1],dimensions[(i*3)+2]], dtype=int)
    for x in range(len(arr)):
        for y in range(len(arr[x])):
            for z in range(len(arr[x][y])):
                #generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
                randomNumber = random.randint(1,9)
                if(randomNumber==4 or randomNumber==5 or randomNumber==6):
                    numToInsert = random.randint(1,100)
                    arr[x][y][z] = numToInsert
                    numToInsert = 0
                    randomNumber = 0
                #print(arr[x][y][z])
                if(arr[x][y][z]!=0):
                    lineToAddInFile="" + str(x) + " " + str(y) + " " + str(z) + " " + str(arr[x][y][z])
                    f.write(lineToAddInFile + '\n')
    tensor_num = tensor_num + 1





#first step: one randomly generated 3D tensor given first set dimensions
#Note: generally if 2/3 elems in a tensor is 0, it can be considered sparse
#approach: 2/3 of the time add in a 0, 1/3 of the time add in an integer from 0 to 100 (use randint to generate num from 1 to 9 inclusive, and depending on where the num is, insert number or not)
#print('dimensions:')
#print(dimensions[0])
#print(dimensions[1])
#print(dimensions[2])
#print('tensor vals')

"""
arr = np.zeros([dimensions[0],dimensions[1],dimensions[2]], dtype=int)
randomNumber = 0
numToInsert = 0
for x in range(len(arr)):
    for y in range(len(arr[x])):
        for z in range(len(arr[x][y])):
            #generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
            randomNumber = random.randint(1,9)
            if(randomNumber==4 or randomNumber==5 or randomNumber==6):
                numToInsert = random.randint(1,100)
                arr[x][y][z] = numToInsert
                numToInsert = 0
            print(arr[x][y][z])

            #lineToAddInFile="" + str(x) + " " + str(y) + " " + str(z) + " " + str(arr[x][y][z])
            #f.write(lineToAddInFile + '\n')

print('dimensions:')
print(dimensions[3])
print(dimensions[4])
print(dimensions[5])
print('tensor vals')
arr = np.zeros([dimensions[3],dimensions[4],dimensions[5]], dtype=int)
randomNumber = 0
numToInsert = 0
for x in range(len(arr)):
    for y in range(len(arr[x])):
        for z in range(len(arr[x][y])):
            #generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
            randomNumber = random.randint(1,9)
            if(randomNumber==4 or randomNumber==5 or randomNumber==6):
                numToInsert = random.randint(1,100)
                arr[x][y][z] = numToInsert
                numToInsert = 0
                randomNumber = 0
            print(arr[x][y][z])

            #lineToAddInFile="" + str(x) + " " + str(y) + " " + str(z) + " " + str(arr[x][y][z])
            #f.write(lineToAddInFile + '\n')
"""
