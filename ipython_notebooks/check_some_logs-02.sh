
resfile=/tmp/tensorflow_logs/copy8-rnn132-f/test_y_and_prediction.62138
awk '{if ($1 != $2) {print "_" $1 "_ _" $2 "_"}}' $resfile  | python class_to_label.py | grep "fi" | grep "en" | sort | uniq -c | sort -nr  | less

