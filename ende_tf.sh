#!/bin/bash
model=model1
corpus=corpus

SL=en
TL=de
pref=spinning-storage/rameshak

/home/rameshak/$pref/software/marian/build/marian \
--model $model/model.npz --type transformer \
--train-sets $corpus/train.BPE.$SL $corpus/train.BPE.$TL \
--max-length 100 \
--vocabs $model/vocab.$SL.yml $model/vocab.$TL.yml \
--mini-batch-fit -w 9000 --mini-batch 1000 --maxi-batch 1000 \
--early-stopping 10 \
--valid-freq 5000 --save-freq 5000 --disp-freq 500 \
--valid-metrics ce-mean-words perplexity translation \
--valid-sets corpus/devset.BPE.$SL corpus/devset.BPE.$TL \
--valid-translation-output $model/valid.bpe.$TL.output --quiet-translation \
--valid-script-path ./validate.sh \
--valid-mini-batch 64 \
--keep-best \
--cost-type=ce-mean-words \
--beam-size 12 --normalize 1 \
--log $model/train.log --valid-log $model/valid.log \
--enc-depth 6 --dec-depth 6 \
--transformer-heads 8 \
--transformer-postprocess-emb d \
--transformer-postprocess dan \
--transformer-dropout 0.1 --label-smoothing 0.1 \
--learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
--optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
--tied-embeddings \
--devices 0 1 2 --seed 999 \
--exponential-smoothing
