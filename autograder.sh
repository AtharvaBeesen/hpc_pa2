#!/bin/bash

#SBATCH --job-name=mpi_results
#SBATCH --output=mpi_results.out
#SBATCH --error=mpi_results.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:05:30

mkdir -p autograderout

MAX_EXECUTION_TIME=300

# Run the tests within the time limit
timeout $MAX_EXECUTION_TIME bash -c '

lscpu
echo "-------------------------------------"

module load openmpi

make clean
make many2many
make allreduce

echo " "
echo "=================== CHECKING TEST FILE ==================="
echo " "

if [ ! -f "autograderdata/inputard_15.txt" ]; then
    echo "No AllReduce 15 Input"
    exit 1
fi

if [ ! -f "autograderdata/inputm2_15.txt" ]; then
    echo "No M2M 15 Input"
    exit 1
fi

if [ ! -f "autograderdata/inputard_24.txt" ]; then
    echo "No AllReduce 24 Input"
    exit 1
fi

if [ ! -f "autograderdata/inputm2_24.txt" ]; then
    echo "No M2M 24 Input"
    exit 1
fi

if [ ! -f "autograderdata/outputard_15.txt" ]; then
    echo "No AllReduce 15 Output"
    exit 1
fi

if [ ! -f "autograderdata/outputm2_15.txt" ]; then
    echo "No M2M 15 Output"
    exit 1
fi

if [ ! -f "autograderdata/outputard_24.txt" ]; then
    echo "No AllReduce 24 Output"
    exit 1
fi

if [ ! -f "autograderdata/outputm2_24.txt" ]; then
    echo "No M2M 24 Output"
    exit 1
fi


echo " "
echo "=================== CORRECTNESS TESTS ==================="
echo " "

echo "##AllReduce 15"
output=$(srun -n 15 ./ard autograderdata/inputard_15.txt autograderout/myoutputard_15.txt)
echo "$output"
if diff autograderdata/outputard_15.txt autograderout/myoutputard_15.txt > /dev/null; then
    echo "AllReduce P=15 Correctness passed!"
else
    echo "AllReduce P=15 Failed"
fi
echo " "

echo "##AllReduce 24"
output=$(srun -n 24 ./ard autograderdata/inputard_24.txt autograderout/myoutputard_24.txt)
echo "$output"

if diff autograderdata/outputard_24.txt autograderout/myoutputard_24.txt > /dev/null; then
    echo "AllReduce P=24 Correctness passed!"
else
    echo "AllReduce P=24 Failed"
fi
echo " "


echo " "
echo "=================================="
echo " "


echo "##Many2Many 15"
output=$(srun -n 15 ./m2m autograderdata/inputm2_15.txt autograderdata/outputm2_15.txt)
echo "$output"
echo " "

echo "##Many2Many 24"
output=$(srun -n 24 ./m2m autograderdata/inputm2_24.txt autograderdata/outputm2_24.txt)
echo "$output"


echo "==================== RUNTIME TESTS ====================="
echo " "
echo "##AllReduce"
output_24=$(srun -n 24 ./ard autograderdata/inputard_24.txt)

custom_time=$(echo "$output_24" | sed -n "s/.*Time taken by custom function: \([0-9.]*\).*/\1/p")
echo "Custom function time: $custom_time"

mpi_time=40
multiplier=1
if (( $(echo "$custom_time < $multiplier * $mpi_time" | bc -l) )); then
        echo "Runtime test passed!"
else
        echo "Runtime test failed"
fi

mpi_time=30
multiplier=1
if (( $(echo "$custom_time < $multiplier * $mpi_time" | bc -l) )); then
        echo "Runtime test passed!"
else
        echo "Runtime test failed"
fi


echo " "
echo "=================================="
echo " "

echo "##Many2Many"

output_24=$(srun -n 24 ./m2m autograderdata/inputm2_24.txt)

custom_time=$(echo "$output_24" | sed -n "s/.*Time taken by custom function: \([0-9.]*\).*/\1/p")
echo "Custom function time: $custom_time"

mpi_time=50
multiplier=1
if (( $(echo "$custom_time < $multiplier * $mpi_time" | bc -l) )); then
        echo "Runtime test passed!"
else
        echo "Runtime test failed"
fi

mpi_time=40
multiplier=1
if (( $(echo "$custom_time < $multiplier * $mpi_time" | bc -l) )); then
        echo "Runtime test passed!"
else
        echo "Runtime test failed"
fi
'
rm -rf autograderout/
if [[ $? -eq 124 ]]; then
    echo "Error: Script execution exceeded $MAX_EXECUTION_TIME seconds."
    exit 1
fi
