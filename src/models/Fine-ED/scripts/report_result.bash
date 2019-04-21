#!/bin/bash

# argument 1: conll evaluation script
# argument 2: result directory
# display result in sorted order
model_numbers=()
fb1_score=()
for entry in `ls $2 | sort -g | egrep -v '.txt'`
do 
  echo $entry
  result=$($1 < $2$entry)
  echo "$result"
  # parse result
  readarray -t y <<<"$result"
  # first element of array contains overall result.
  model_numbers+=($entry)
  fb1_score+=(${y[1]:62:5})
done
for ((i=0;i<${#model_numbers[@]};++i)); do
      printf "Model no. %s performance is %s\n" "${model_numbers[i]}" "${fb1_score[i]}"
done
echo "Maximum F1 score:"
printf "%s\n" "${fb1_score[@]}" | datamash max 1
