
mkdir -p recipes
rm recipes/*.recipe

# Sample file location:
# /teamwork/t40511_asr/c/speecon-fi/child/CHILD1FI/BLOCK00/SES001/SC001002.FI0
audiodir="/teamwork/t40511_asr/c/speecon-fi/child/CHILD1FI/"
find `pwd`/raw_labels/ -name "*.lab" | while read labfile; do
    base=`basename $labfile .lab`
    block=`echo $base | sed -r 's/..(..).*/\1/g'`
    sess=`echo $base | sed -r 's/..(...).*/\1/g'`
    audiofile="$audiodir/BLOCK$block/SES$sess/$base.FI0"
    if [ -f "$audiofile" ]; then
	echo "audio=$audiofile transcript=$labfile age=?" >> recipes/$sess.recipe
    fi
done
