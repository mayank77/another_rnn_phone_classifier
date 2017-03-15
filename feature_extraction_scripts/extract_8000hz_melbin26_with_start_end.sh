#!/bin/bash
#set -e
set -x

sptk="/usr/local/bin";
sptk="/teamwork/t40511_asr/Modules/opt/sptk/3.8/bin"

workdir=/dev/shm/siak-feat-extract/`date +"%s-%N"`-$RANDOM-$RANDOM-$RANDOM

mkdir -p $workdir

# Get the binary stdinput into a tmp memory file:

# For speecon data files & other headerless signed-integer 16000Hz files:
#sox  -t raw -e signed-integer -r 16000 -b 16 $1 -t raw -e floating-point -r 8000 -b 32  $workdir/data.tmp.rawinput  norm -20

cat $1 | $sptk/bcut +s -s $3 -e $4 >  $workdir/data.tmp.cut_rawinput

# For headerless signed-integer 8000Hz files:
sox  -t raw -e signed-integer -r 8000 -b 16 $workdir/data.tmp.cut_rawinput -t raw -e floating-point -r 8000 -b 32  $workdir/data.tmp.rawinput  norm -20

# For general float data:
# sox  -t raw -e floating-point -r 8000 -b 16 $1 -t raw -e floating-point -r 8000 -b 32  $workdir/data.tmp.rawinput  norm -20


#sox $workdir/data.tmp.wav -t raw -e floating-point -r 8000 -b 32  $workdir/data.tmp.rawinput  norm -20

fs=8000;
samplingkhz=8
max_utterance_length_s=10;
max_packet_length_s=1;
datatype_length=4;
frame_step_samples=64 #$(( 128 * fs / 16000 ));
frame_length_samples=256 #$(( 400 * fs / 16000 ));

window_length_samples=256;
min_lpc_determinant=0.000001;

fft_length=256;

err=0

# -o 2 would give us log(f0) but we'll take f0 instead:
$sptk/pitch -a 1 -o 0 -s 8 -p $frame_step_samples -L 120 -H 300 $workdir/data.tmp.rawinput > $workdir/data.tmp.pitch8

fft1len=512
fft2len=128
speclen=$(( fft2len / 2 + 1))
numsmoothingceps=64
numsynthceps=25
nummelbanks=36

$sptk/frame -p $frame_step_samples -l $frame_length_samples < $workdir/data.tmp.rawinput | $sptk/window -l $frame_length_samples -L $fft1len >  $workdir/data.tmp.windowed
$sptk/fftr -A -H –l $fft1len $workdir/data.tmp.windowed | x2x +fa258 >  $workdir/data.tmp.spec.ascii

python3 apply_melbank_25x129.py $workdir/data.tmp.spec.ascii  $workdir/data.tmp.melbinspec.ascii


#wc -l $workdir/data.tmp.melbinspec.ascii  > /tmp/mel_rows
#x2x +fa $workdir/data.tmp.pitch8 | wc -l > /tmp/pitch_rows

$sptk/x2x +af $workdir/data.tmp.melbinspec.ascii > $workdir/data.tmp.melbinspec
$sptk/merge +f -s $nummelbanks -L $nummelbanks -l 1 $workdir/data.tmp.melbinspec <  $workdir/data.tmp.pitch8 > $2

#x2x +fa37 $2 > /tmp/feattest



rm $workdir/data.tmp.*
rmdir $workdir
#echo workdir: $workdir 1>&2
