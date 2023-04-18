#!/bin/sh
python3 -u pretrain.py --exp_name=first_try --batch_size=128 --epochs=32 --learning_rate=0.1 > script_log/logs_pretrain_exp.txt