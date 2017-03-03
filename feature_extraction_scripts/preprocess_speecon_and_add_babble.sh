#!/bin/bash
#
#  $1 scp file (train/eval/valid) 
#  $2 speaker & sentence id (eg. 04682001-0003-598-1)
#  $3 target location for a wav file
#  $4 babble normalisation
#  $5 speech normalisation

#/teamwork/t40511_asr/p/digitala/models1/extract_wav.py $1 $2 | sox -t wav -r 16000 -c 1 -b 16 - $3

# Hard coded babble noise file location:

noisefile=/teamwork/t40511_asr/c/noisex92/audio_wav/babble.wav
noisedursamples=3763687

tmpspeechfile=/dev/shm/noise.wav
tmpnoisefile=/dev/shm/speech.wav

/teamwork/t40511_asr/p/digitala/models1/extract_wav.py $1 $2 | sox -t wav -r 16000 -c 1 -b 16 - $tmpspeechfile norm $5

speechdursamples=`soxi -s tmpspeechfile`

sox $noisefile -r 16000 -c 1 -b 16 $tmpnoisefile trim `shuf -i 0-$(( $noisedursamples - $speechdursamples )) -n 1` $speechdursamples norm $4

sox -m $tmpnoisefile $tmpspeechfile $3

rm $tmpspeechfile
rm $tmpnoisefile
