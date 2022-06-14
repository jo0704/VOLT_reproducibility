#!/bin/bash


output_path=$"../VOLT"/output
#mkdir $output_path

#number gpu = 4
data="bpeout_1-2k"
max_token=9600
max_epoch=100
update_freq=8
lr=0.0005
total_word=100000

python3 fairseq/fairseq_cli/train.py --num-workers 8 de_en \
--arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr $lr --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
--dropout 0.3 --weight-decay 0.0 --update-freq $update_freq \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_token \
--fp16 --max-epoch $max_epoch --keep-last-epochs 5 \
--save-dir de_en/checkpoints

# transformer_vaswani_wmt_en_fr_big