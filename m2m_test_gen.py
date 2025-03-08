#!/usr/bin/env python3
import argparse
import os

# Set up argument parser to get the number of processors
parser = argparse.ArgumentParser(description="Generate Many-to-Many test input file")
parser.add_argument("P", type=int, help="Number of processors")
args = parser.parse_args()
P = args.P

# Fixed problem size: approximate data size to be sent to each processor
N = 10000000  
increment = 2  # Arbitrary increment for generating send counts and data

# Create the output directory "testingdata" if it doesn't exist
os.makedirs("testingdata", exist_ok=True)
filename = f"testingdata/inputm2_{P}.txt"

with open(filename, "w") as f:
    # Write the number of processors
    f.write(f"{P}\n")
    
    # Write the send count arrays: one line per processor, each with P integers.
    for i in range(P):
        counts = [str(N + j * increment) for j in range(P)]
        f.write(" ".join(counts) + "\n")
    
    # Write the send data arrays: one line per processor.
    # For each processor, for each destination, generate an array of the given count.
    for i in range(P):
        data_line = []
        for j in range(P):
            count = N + j * increment
            # For example, fill with the processor id 'i'
            data_line.extend([str(i)] * count)
        f.write(" ".join(data_line) + "\n")
        
print(f"Many-to-Many test input file generated: {filename}")
