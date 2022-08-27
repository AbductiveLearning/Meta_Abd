#!/bin/bash

echo "Trial 1"
python -u run_sum.py 2>&1 | tee models/log_1.txt
echo "Trial 2"
python -u run_sum.py 2>&1 | tee models/log_2.txt
echo "Trial 3"
python -u run_sum.py 2>&1 | tee models/log_3.txt
echo "Trial 4"
python -u run_sum.py 2>&1 | tee models/log_4.txt
echo "Trial 5"
python -u run_sum.py 2>&1 | tee models/log_5.txt
