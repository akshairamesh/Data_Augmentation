#!/bin/bash

corpus=corpus

SL=en
TL=de
pref=spinning-storage/rameshak

#This is for en-ta 
echo 'started preprocessing the data for training the model'
python3 /home/rameshak/$pref/software/OpenNMT-py/preprocess.py -train_src $corpus/train.BPE.$SL -train_tgt $corpus/train.BPE.$TL -valid_src $corpus/devset.BPE.$SL -valid_tgt $corpus/devset.BPE.$TL \
-save_data corpus1/model_data \
--src_vocab_size 16000 \
--tgt_vocab_size 16000


echo 'preprocessed data saved in corpus1/model_data'
python3 /home/rameshak/$pref/software/OpenNMT-py/train.py -data corpus1/model_data -save_model model_ta_en/brnn  \
--word_vec_size 512 --encoder_type brnn --decoder_type rnn --rnn_size 1024 \
--enc_layers 1 --dec_layers 1 --label_smoothing 0.2 --batch_type tokens --normalization sents --batch_size 1000 \
--valid_steps 2000 --learning_rate 0.0005 --average_decay 1e-4 --optim adam --early_stopping 10 --dropout 0.2 \
--world_size 1 --gpu_ranks 0 --save_checkpoint_steps 20000 --master_port 9200
