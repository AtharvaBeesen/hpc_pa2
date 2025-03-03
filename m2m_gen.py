import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate input file for processors.')
parser.add_argument('P', type=int, help='number of processors')

# Parse the arguments
args = parser.parse_args()
P = args.P

N = 1000000		# approx data to be sent to every other processor.
# P = 24 						# number of processors  
increment = 2 				# keep it here for now 
filename = f"autograderdata/inputm2_{P}.txt"  # name of the input file to be generated
outname = f"autograderdata/outputm2_{P}.txt"  # name of the output file to be generated

# open a file with filename 
f = open(filename, "w")

# open a file with filename 
f = open(filename, "w")

# write P as the first line of the file
f.write(str(P) + "\n")

array = [[0 for _ in range(P)] for _ in range(P)]

# Now write the sendcounts arrays 
for i in range(0, P):
	curr = N
	for j in range(0, P):
		f.write(str(curr) + " ")
		array[i][j] = curr
		curr += increment
	f.write("\n")

print("M2M " + str(P) + " Sendcount Generated...")

# Now write the send_data arrays 
for i in range(0, P):
	curr = N
	data = 0
	for j in range(0, P):
		data_line = " ".join([str(data)] * curr)
		f.write(data_line + " ")
		curr += increment
		data += increment
	f.write("\n")
	
f.close()
print("M2M " + str(P) + " Input Generated...")

o = open(outname, "w")
for i in range(0, P):
	data = i * increment
	for j in range(0, P):
		data_line = " ".join([str(data)] * array[j][i])
		o.write(data_line + " ")
	o.write("\n")
print("M2M " + str(P) + " Output Generated...")
