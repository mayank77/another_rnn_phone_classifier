rm raw_labels/*/*.raw_label

phonecount=`wc -l kaldi_labels/phones.ctm | cut -d ' ' -f 1`
count=0

cat kaldi_labels/phones.ctm  | while read l; do 
    speaker=`echo $l | sed -r 's/(.*)\-ch0.*/\1/g'`
    file=`echo $l | sed -r 's/(.*)\-ch0\-(...).*/\2/g'`
    content=`echo $l | sed -r 's/.*\-ch0\-... 1 (.*)/\1/g'`
    #echo "$speaker --- $file --- $content"
    #mkdir -p raw_labels/$speaker
    echo "$content" >> raw_labels/$speaker/$speaker$file.raw_label
    export count=$(( count + 1 ))
    echo -ne "\r$count/$phonecount"
done
