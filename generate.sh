#!/bin/bash


output_path=$"../VOLT"/output
# mkdir $output_path

#number gpu = 4
data="bpeout_1-2k"
max_token=9600
max_epoch=100
update_freq=8
lr=0.0005
total_word=100000

python3 fairseq/fairseq_cli/generate.py de_en \
--path checkpoint.avg5.pt \
--beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > gen1.out