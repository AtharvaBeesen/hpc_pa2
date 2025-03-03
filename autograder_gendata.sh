#!/bin/bash

echo "=================== GENERATING TEST FILE ==================="
echo " "
mkdir -p autograderdata

if [ ! -f "inputard_15.txt" ]; then
    python3 ard_gen.py 15
fi
echo " "

if [ ! -f "inputm2_15.txt" ]; then
    python3 m2m_gen.py 15
fi
echo " "

if [ ! -f "inputard_24.txt" ]; then
    python3 ard_gen.py 24
fi
echo " "

if [ ! -f "inputm2_24.txt" ]; then
    python3 m2m_gen.py 24
fi

echo " "
exit 1
