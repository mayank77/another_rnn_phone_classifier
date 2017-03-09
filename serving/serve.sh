#!/bin/bash
# A shell wrapper for the python thingy:

PATH=/l/opt/anaconda3/bin:/usr/local/bin/:$PATH

source activate tensorflow012

python3 ./serve_bidir_lstm_384x6.py \
    /home/condadnn/another_rnn_phone_classifier/serving/ports/portnr \
    /home/condadnn/another_rnn_phone_classifier/models/bidir_lstm_6x384_checkpoint_16000/model2.ckpt-16000 \
    /home/condadnn/another_rnn_phone_classifier/models/bidir_lstm_6x384_checkpoint_16000/traindata.mean \
    /home/condadnn/another_rnn_phone_classifier/models/bidir_lstm_6x384_checkpoint_16000/traindata.std \
    /home/condadnn/another_rnn_phone_classifier/models/bidir_lstm_6x384_checkpoint_16000/lsq_weights \

