#!/bin/bash
echo "Generating testing input files..."

# Create the testingdata directory if it doesn't exist
mkdir -p testingdata

for p in 4 8 16 24
do
    if [ ! -f "testingdata/inputard_${p}.txt" ]; then
        python3 ard_test_gen.py ${p}
    fi

    if [ ! -f "testingdata/inputm2_${p}.txt" ]; then
        python3 m2m_test_gen.py ${p}
    fi
done

echo "Testing input files generated in the folder 'testingdata'."
