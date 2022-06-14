# VOLT_reproducibility
This repository contains the scripts we used to reproduce the results in: Jingjing Xu, Hao Zhou, Chun Gan, Zaixiang Zheng, and Lei Li. 2021. Vocabulary learning via optimal transport for neural machine translation. In Proceedings of ACL 2021

The preprocessing scripts with VOLT were adapted from the readme of the official VOLT repository and https://github.com/Jingjing-NLP/VOLT/blob/master/examples/fairseq_command/train_ende.sh for the training process

## Requirements

git clone https://github.com/Jingjing-NLP/VOLT/

git clone https://github.com/moses-smt/mosesdecoder.git

git clone https://github.com/rsennrich/subword-nmt.git

git clone https://github.com/pytorch/fairseq \
cd fairseq \
pip install --editable ./

## Data

All data files have been fetched here: https://drive.google.com/drive/folders/1CkcpPu7ovPuvLpCbl1cBLnx510mT0Q2W

## Preprocessing
Preprocessing with the VOLT vocabulary involves three different scripts which need to be run in the order specified below. Examples on how to run the scripts are given based on the fr-en language pair (initial vocabulary size 100k). If you are using these scripts, make sure to adapts the data paths.

```preprocessing_bpe.sh```: ./preprocess_bpe.sh 100000 fr en \
Segment text into subword units with subword-nmt \
After running this script, we get the BPE vocab size in the train splits for fr-en to check the results against the vocab size in the original VOLT paper

```preprocessing_VOLT.sh```: .preprocess_VOLT.sh 100000 fr en \
apply VOLT algorithm to BPE splits \
After running this script, we get the VOLT vocab size in the train splits for fr-en to check the results against the vocab size in the original VOLT paper

```preprocess.sh```: ./preprocess.sh \
prepare files for training


## Training
Training was executed with a GPU on Google Cloud using fairseq, in the following order:

```train.sh```: uses one GPU 

```checkpoints.sh```: averages the params of 5 model checkpoints

```generate.sh```: remove bpe splits

```bash fairseq/scripts/compound_split_bleu.sh score.out```: computes BLEU score
