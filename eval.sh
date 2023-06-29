#!/bin/sh
python3 -u model/evaluate.py --dataset_name=indosum_test_32k > result/indosum_test_32k.csv
python3 -u model/evaluate.py --dataset_name=liputan6_test_32k > result/liputan6_test_32k.csv
python3 -u model/evaluate.py --dataset_name=xlsum_32k_test > result/xlsum_32k_test.csv
python3 -u model/generate_iter.py --dataset_name=xlsum_32k --num_files=2 > result/xlsum_32k.txt
python3 -u model/generate_iter.py --dataset_name=indosum_32k --num_files=4 > result/indosum_32k.txt
python3 -u model/generate_iter.py --dataset_name=liputan6_32k --num_files=11 > result/liputan6_32k.txt