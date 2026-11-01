#!/bin/bash
# Generate the MOFs of one ToBaCCo template (net). usage: bash run_one.sh TEMPLATE   (run inside the tob_test folder)
cd "$(dirname "$0")"; mkdir -p logs
timeout 120 python3 -c "import sys; sys.setrecursionlimit(10000); import tobacco; tobacco.run_template('$1')" > logs/$1.log 2>&1
echo "$1 $?" >> done.txt
