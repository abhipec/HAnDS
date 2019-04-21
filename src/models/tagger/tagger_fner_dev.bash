conll_file=$(mktemp /tmp/conll.XXXXXXXXX)
sentence_file=$(mktemp /tmp/sentence.XXXXXXXXX)
tagger_result_file=$(mktemp /tmp/tagger_result.XXXXXXXXX)
output_dir=../../results/Fine-ED/tagger/
mkdir -p $output_dir
final_result_conll_file=$output_dir/fner_dev.conll
python3 Fine-ED/src/json_to_conll.py json_data ../../datasets/1k-WFB-g/fner_dev.json $conll_file
python3 ../../utils/conll_to_plain_text.py $conll_file $sentence_file
python ../../utils/tagger/tagger.py --model ../../utils/tagger/models/english/ --input $sentence_file  --output $tagger_result_file
sed -i -- 's/B-ORG/B-E/g' $tagger_result_file
sed -i -- 's/I-ORG/I-E/g' $tagger_result_file
sed -i -- 's/B-PER/B-E/g' $tagger_result_file
sed -i -- 's/I-PER/I-E/g' $tagger_result_file
sed -i -- 's/B-LOC/B-E/g' $tagger_result_file
sed -i -- 's/I-LOC/I-E/g' $tagger_result_file
sed -i -- 's/B-MISC/B-E/g' $tagger_result_file
sed -i -- 's/I-MISC/I-E/g' $tagger_result_file

python3 ../../utils/tagger_output_to_conll.py $tagger_result_file $tagger_result_file.iob

paste $conll_file $tagger_result_file.iob  > $final_result_conll_file
sed -i -- 's/\t/ /g' $final_result_conll_file
sed -i -- 's/  / /g' $final_result_conll_file
sed -i -- 's/^ //g' $final_result_conll_file
