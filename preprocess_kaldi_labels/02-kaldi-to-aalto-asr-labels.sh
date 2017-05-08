
find raw_labels/ -name "*.raw_label" | while read f; do
    python kaldi_to_asr_labels.py $f > `echo $f | sed -r 's/\.raw_label/.lab/g'`
done
