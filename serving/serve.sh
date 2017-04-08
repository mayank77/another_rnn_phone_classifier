#!/bin/bash
# A shell wrapper for the python thingy:

PATH=/l/opt/anaconda3/bin:/usr/local/bin/:$PATH

source activate tensorflow012

python3 ./serve_bidir_lstm_512x5-melbin36.py \
    /home/condadnn/another_rnn_phone_classifier/serving/ports/portnr \
    /home/condadnn/another_rnn_phone_classifier/models/rnn512-e/model2.ckpt-48055 \
    /home/condadnn/another_rnn_phone_classifier/models/rnn512-e/traindata.mean \
    /home/condadnn/another_rnn_phone_classifier/models/rnn512-e/traindata.std \
    /home/condadnn/another_rnn_phone_classifier/models/rnn512-e/testscores-48055/lsq_weights

