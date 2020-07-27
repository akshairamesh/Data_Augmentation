#!/bin/bash

pref=spinning-storage/rameshak
opennmt=/home/rameshak/$pref/software/OpenNMT-py
model=model
SL=en
TL=de
#sys.path("/home/rameshak/.local/lib/python3.7/site-packages/")
#we have 1 test set

tes=corpus1/testset2014.$SL   
ref=corpus1/testset2014.$TL     

nvidia-smi 

evaluation=evaluation

#mkdir $evaluation -p

# Decoding (translation)
python3 $opennmt/translate.py -model /home/rameshak/spinning-storage/rameshak/wmt2020/scripts/ende/model1/base_tf_6lyr_32k_step_70000.pt -src $tes -output $evaluation/pred_6lyr_32k_tst1.txt -replace_unk -verbose

#Evaluation
#/home/rameshak/spinning-storage/rameshak/software/scripts/multeval/multeval.sh eval --refs $ref --hyps-baseline $evaluation/pred_wmt.txt --meteor.language en > $evaluation/results_wmt
#Evaluation -- measuring bleu
/home/rameshak/$pref/software/scripts/generic/multi-bleu.perl $ref < $evaluation/pred_6lyr_32k_tst1.txt > $evaluation/results_6lyr_32k_tst1

tes=corpus1/testset2015.$SL   
ref=corpus1/testset2015.$TL 

python3 $opennmt/translate.py -model /home/rameshak/spinning-storage/rameshak/wmt2020/scripts/ende/model1/base_tf_6lyr_32k_step_70000.pt -src $tes -output $evaluation/pred_6lyr_32k_tst2.txt -replace_unk -verbose

#Evaluation
#/home/rameshak/spinning-storage/rameshak/software/scripts/multeval/multeval.sh eval --refs $ref --hyps-baseline $evaluation/pred_wmt.txt --meteor.language en > $evaluation/results_wmt
#Evaluation -- measuring bleu
/home/rameshak/$pref/software/scripts/generic/multi-bleu.perl $ref < $evaluation/pred_6lyr_32k_tst2.txt > $evaluation/results_6lyr_32k_tst2


