#!/usr/bin/env python3
import argparse
import os

# Set up argument parser to get the number of processors
parser = argparse.ArgumentParser(description="Generate Allreduce test input file")
parser.add_argument("P", type=int, help="Number of processors")
args = parser.parse_args()
P = args.P

# Fixed problem size: number of elements per processor
N = 10000000  
increment = 2  # Arbitrary increment to generate values

# Create the output directory "testingdata" if it doesn't exist
os.makedirs("testingdata", exist_ok=True)
filename = f"testingdata/inputard_{P}.txt"

with open(filename, "w") as f:
    # Write the number of processors and the number of elements
    f.write(f"{P}\n")
    f.write(f"{N}\n")
    # Write the local arrays: each processor gets one line with N space-separated integers
    for i in range(P):
        values = [str(i + j * increment) for j in range(N)]
        f.write(" ".join(values) + "\n")
        
print(f"Allreduce test input file generated: {filename}")
