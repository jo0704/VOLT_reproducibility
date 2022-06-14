#!/bin/bash


output_path=$"../VOLT"/output
# mkdir $output_path

#number gpu = 4
data="../bpeout_1-2k"
max_token=9600
max_epoch=100
update_freq=8
lr=0.0005
total_word=100000

python3 fairseq/scripts/average_checkpoints.py \
--inputs /mnt/c/Users/jolll/OneDrive/Desktop/Seminar/VOLT/de_en/checkpoints \
--num-epoch-checkpoints 5 \
--output checkpoint.avg5.pt