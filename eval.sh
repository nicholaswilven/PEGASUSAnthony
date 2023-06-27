#!/bin/sh
python3 -u model/evaluate.py --dataset_name=indosum_test_32k
python3 -u model/evaluate.py --dataset_name=liputan6_test_32k
python3 -u model/evaluate.py --dataset_name=xlsum_32k_test
python3 -u model/generate_iter.py --dataset_name=indosum_test_32k > result/indosum_test_32k.txt
python3 -u model/generate_iter.py --dataset_name=liputan6_test_32k > result/liputan6_test_32k.txt
python3 -u model/generate_iter.py --dataset_name=xlsum_32k_test > result/xlsum_32k_test.txt