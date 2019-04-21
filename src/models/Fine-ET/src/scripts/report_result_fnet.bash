#!/bin/bash

# argument 1: result directory
# argument 2: Json file directory
# argument 3: if 0 normal result, if 1 top level result only
# argument 4: Label patch file
# display result in sorted order
model_numbers=()
f1_score=()
for entry in `ls $1 | sort -g | egrep -v '.txt'`
do 
  echo $entry
  result=$(python Fine-ET/src/report_result_fnet.py $2 ${1}$entry $3 $4)
  echo "$result"
  readarray -t y <<<"$result"
  # first element of array contains overall result.
  model_numbers+=($entry)
  f1_score+=(${y[5]:12:6})
done
for ((i=0;i<${#model_numbers[@]};++i)); do
      printf "Model no. %s performance is %s\n" "${model_numbers[i]}" "${f1_score[i]}"
done
echo "Maximum F1 score:"
printf "%s\n" "${f1_score[@]}" | datamash max 1
