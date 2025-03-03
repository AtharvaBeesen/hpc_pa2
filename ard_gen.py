import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate input file for processors.')
parser.add_argument('P', type=int, help='number of processors')

# Parse the arguments
args = parser.parse_args()
P = args.P

N = 10000000	# length of each array stored locally
increment = 2 				# keep it this for now 
filename = f"autograderdata/inputard_{P}.txt"  # name of the input file to be generated
outname = f"autograderdata/outputard_{P}.txt"  # name of the output file to be generated

# open a file with filename 
f = open(filename, "w")

# write P as the first line of the file
f.write(str(P) + "\n")

# write N as the second line of the file 
f.write(str(N) + "\n")

# Now write the local arrays 
totalsum = 0
for i in range(0, P):
	totalsum += i
	curr = i
	for j in range(0, N):
		f.write(str(curr) + " ")
		curr += increment
	f.write("\n")
print("AllReduce " + str(P) + " Input Generated...")

o = open(outname, "w")

for j in range(0, N):
	ans = totalsum + j*increment*P
	o.write(str(ans) + " ")
o.write("\n")
print("AllReduce " + str(P) + " Output Generated...")
