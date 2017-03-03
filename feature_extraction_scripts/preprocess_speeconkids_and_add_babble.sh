#!/bin/bash
#
#  $1 scp file (train/eval/valid) 
#  $2 target location for a wav file
#  $3 output rate (resample!)
#  $4 babble normalisation
#  $5 speech normalisation

#/teamwork/t40511_asr/p/digitala/models1/extract_wav.py $1 $2 | sox -t wav -r 16000 -c 1 -b 16 - $3

# Hard coded babble noise file location:

noisefile=/teamwork/t40511_asr/c/noisex92/audio_wav/babble.wav
noisedursamples=3763687

tmpspeechfile=/dev/shm/speech.wav
tmpnoisefile=/dev/shm/noise.wav

sox -t raw -r 16000 -b 16 -e signed-integer $1 -t wav -r 16000 -c 1 -b 16 --encoding signed-integer $tmpspeechfile speed $3 norm $5

speechdursamples=`soxi -s $tmpspeechfile`

sox $noisefile -r 16000 -c 1 -b 16 $tmpnoisefile trim `shuf -i 0-$(( $noisedursamples - $speechdursamples )) -n 1`s ${speechdursamples}s norm $4

sox -m $tmpnoisefile $tmpspeechfile -t raw -r 8000 -c 1 -b 16 --encoding signed-integer $2

rm $tmpspeechfile
rm $tmpnoisefile


