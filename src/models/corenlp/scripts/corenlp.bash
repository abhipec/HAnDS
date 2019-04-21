test_json_file=$1
stanford_ner_directory=$2
result_file=$3

#Usage example: bash scripts/corenlp.bash /hdd3/FgER/datasets/1k-WFB-g/fner_test.json /hdd1/fner/utils/stanford-ner-2016-10-31/ /hdd3/FgER/results/Fine-ED/corenlp/fner_test.conll

python src/corenlp_ner_tagger.py $2 $1 $result_file
