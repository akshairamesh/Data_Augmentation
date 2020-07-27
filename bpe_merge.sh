#!/bin/bash

SL=en
TL=de
corpus=corpus1
#we have 1 test set


pref=spinning-storage/rameshak
~/$pref/software/subword_nmt/learn_bpe.py --input $corpus/train.$SL -s 4000 -o $corpus/bpe.$SL.32k &
~/$pref/software/subword_nmt/learn_bpe.py --input $corpus/train.$TL -s 4000 -o $corpus/bpe.$TL.32k &
wait

~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$SL.32k < $corpus/train.$SL > $corpus/train.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$TL.32k < $corpus/train.$TL > $corpus/train.BPE.$TL &

~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$SL.32k < $corpus/testset2014.$SL > $corpus/testset2014.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$TL.32k < $corpus/testset2014.$TL > $corpus/testset2014.BPE.$TL &

~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$SL.32k < $corpus/testset2015.$SL > $corpus/testset2015.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$TL.32k < $corpus/testset2015.$TL > $corpus/testset2015.BPE.$TL &


~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$SL.32k < $corpus/devset.$SL > $corpus/devset.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c $corpus/bpe.$TL.32k < $corpus/devset.$TL > $corpus/devset.BPE.$TL &

wait
echo 'Byte Pair Encoding has been completed...'


