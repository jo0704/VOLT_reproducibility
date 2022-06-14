#!/bin/bash

#adapt data path
output_path="X-EN/fr_en/bpe/volt/100000/output"
mkdir $output_path

#adapt data path
data="X-EN/fr_en/bpe/volt/100000"
max_token=9600
max_epoch=100
update_freq=8
lr=0.0005
total_word=100000

python3 fairseq/fairseq_cli/preprocess.py \
--source-lang fr --target-lang en \
--trainpref $data/train \
--validpref $data/valid \
--testpref $data/test \
--destdir $output_path \
--nwordssrc $total_word --nwordstgt $total_word \
--joined-dictionary \
--workers 20
