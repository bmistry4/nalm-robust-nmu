# Loops through NAU-NRU log files and prints out if a run's final extrap loss is <1e-5 and it the NRU weights are [1,-1]
# or [-1,1] implying they are the correct weights

import os
import sys

# path to log folder
path = str(sys.argv[1])
pass_counter = 0
files_counter = 0

with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".out") and entry.is_file():
            # print(entry.name, entry.path)
            with open(entry.path, 'r') as file:
                files_counter += 1
                lines = file.readlines()

                last_line = lines[-1]
                success_result = last_line.split()[-1]
                if success_result == 'True':
                    print(entry.name, "extrap success")
                    pass_counter += 1
                # strip to remove new list char
                if lines[-3].strip() == 'tensor([[ 1., -1.]])' or lines[-3].strip() == 'tensor([[-1., 1.]])':
                    print(entry.name, "nru weight match")

print("Successes (raw): ", pass_counter)
print("Success (%)", 100 * pass_counter / files_counter)
