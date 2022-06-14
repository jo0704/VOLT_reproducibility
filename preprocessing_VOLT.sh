#!/usr/bin/env bash
# -*- coding: utf-8 -*-

size=$1 # 30000 # the size of BPE
src=$2 # es or de or it
tgt=$3 # en
run_volt="${4:-1}" # set to true

echo "Will run VOLT = $run_volt"

data="X-EN/${src}_${tgt}/"
# bpe_volt="$data/bpe/volt"
bpe="X-EN/${src}_${tgt}/bpe"
# mkdir -p "$bpe/$size"

if [ ${run_volt} -eq "1" ]; then
    
    echo "Running VOLT..."
    echo ""

    mkdir -p "$bpe/volt/$size"

    # learn new vocab with VOLT
    python3 ot_run.py \
        --source_file $bpe/$size/train.$src --target_file $bpe/$size/train.$tgt \
        --token_candidate_file $bpe/code \
        --vocab_file $bpe/volt/$size/vocab \
        --max_number 10000 \
        --interval 1000  \
        --loop_in_ot 500 \
        --tokenizer subword-nmt \
        --size_file $bpe/volt/$size/size

    echo "#version: 0.2" > $bpe/volt/$size/vocab.seg # add version info
    cat $bpe/volt/$size/vocab >> $bpe/volt/$size/vocab.seg

    echo "Applying optimized BPE through VOLT..."
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/train.$src > $bpe/volt/$size/train.$src
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/train.$tgt > $bpe/volt/$size/train.$tgt
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/dev.$src > $bpe/volt/$size/valid.$src
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/dev.$tgt > $bpe/volt/$size/valid.$tgt
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/test.$src > $bpe/volt/$size/test.$src
    python3 subword-nmt/subword_nmt/apply_bpe.py -c $bpe/volt/$size/vocab.seg < $data/test.$tgt > $bpe/volt/$size/test.$tgt

#     # get vocab size
    echo ""
    vocab_size=$(cat $bpe/volt/$size/train.$src $bpe/volt/$size/train.$tgt | tr ' ' '\n' | sort | uniq | wc -l)
    echo "VOLT vocab size in train splits for $src-$tgt (BPE symbols = $size): $vocab_size"

fi