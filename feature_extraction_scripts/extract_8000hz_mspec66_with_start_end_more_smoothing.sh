#!/bin/bash
#set -e

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


fs=8000;
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
numsmoothingceps=32
numsynthceps=25

$sptk/frame -p $frame_step_samples -l $frame_length_samples < $workdir/data.tmp.rawinput | $sptk/window -l $frame_length_samples -L $fft1len | $sptk/mcep -a 0.31 -l $fft1len -m $numsmoothingceps | c2sp -m $numsmoothingceps -o 2 -l $fft2len > $workdir/data.tmp.mspec8

$sptk/merge +f -s $speclen -L $speclen -l 1 $workdir/data.tmp.mspec8 <  $workdir/data.tmp.pitch8 > $2

cat $2 > /tmp/testfeat

if [ "$5" != "" ]; then 
    # In theory, these commands will resynthesise the utterance:
    $sptk/gcep -c 2.5 -q 4 -e 0.001 -l $fft2len -m $numsynthceps < $workdir/data.tmp.mspec8 > $workdir/data.tmp.mcep8
    $sptk/excite -p 80  < $workdir/data.tmp.pitch8 | $sptk/mlsadf -p 80 -m $numsynthceps -a 0.31 $workdir/data.tmp.mcep8 > $workdir/data.tmp.syn8
    sox -t raw -r 8000 -e floating-point -b 32 $workdir/data.tmp.syn8 -b 16 $5 norm -10
fi



rm $workdir/data.tmp.*
rmdir $workdir
#echo workdir: $workdir 1>&2
