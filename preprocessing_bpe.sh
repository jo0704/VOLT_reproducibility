#!/usr/bin/env bash
# -*- coding: utf-8 -*-

size=$1 # 30000 # the size of BPE
src=$2 # es or de or it
tgt=$3 # en
run_volt="${4:-0}" # false by default

echo "Will run VOLT = $run_volt"

#adapt paths here
data="/${src}_${tgt}/"
bpe="/${src}_${tgt}/bpe"

# #subword-nmt style:
mkdir -p "$bpe/$size"

echo "Learning initial BPE..."

cat $data/train.$src $data/train.$tgt > $bpe/all_train.$src.$tgt
# get initial bpe with default size
python3 subword-nmt/subword_nmt/learn_bpe.py -s $size < $bpe/all_train.$src.$tgt > $bpe/code

echo "$bpe"

echo "Applying initial BPE..."
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/train.$src > $bpe/$size/train.$src
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/train.$tgt > $bpe/$size/train.$tgt
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/dev.$src > $bpe/$size/valid.$src
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/dev.$tgt > $bpe/$size/valid.$tgt
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/test.$src > $bpe/$size/test.$src
python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/code < $data/test.$tgt > $bpe/$size/test.$tgt

# # get vocab size
echo ""
vocab_size=$(cat $bpe/$size/train.$src $bpe/$size/train.$tgt | tr ' ' '\n' | sort | uniq | wc -l)
echo "BPE vocab size in train splits for $src-$tgt (BPE symbols = $size): $vocab_size"
# echo ""
