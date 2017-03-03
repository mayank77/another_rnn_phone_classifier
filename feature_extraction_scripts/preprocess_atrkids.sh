#!/bin/bash
#
#  $1 scp file (train/eval/valid) 
#  $2 speaker & sentence id (eg. 04682001-0003-598-1)
#  $3 target location for a wav file
#
#

#/teamwork/t40511_asr/p/digitala/models1/extract_wav.py $1 $2 | sox -t wav -r 16000 -c 1 -b 16 - $3

/teamwork/t40511_asr/p/digitala/models1/extract_wav.py $1 $2 | sox -t wav -r 16000 -c 1 -b 16 - $3

