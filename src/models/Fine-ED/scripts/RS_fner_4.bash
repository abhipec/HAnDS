data_directory=../../model_training_data/Fine-ED/RS_fner_4
ckpt_directory=$data_directory/ckpts
dataset_file=../../datasets/sampled_datasets/RS_fner_4.json.gz
mkdir -p $ckpt_directory

rnn_hidden_neurons=100
keep_prob=0.5
learning_rate=0.001
batch_size=500
char_embedding_size=50
lstm_layers=1
epochs=2
save_checkpoint_after=200000

glove_vector_file_path=/hdd1/word_vectors/glove.42B.300d/glove.42B.300d.txt

#echo "Converting training data to ConLL format."
#python Fine-ED/src/json_to_conll.py json_data $dataset_file $data_directory/train.conll


#echo "Converting evaluation datasets to ConLL format."
#python Fine-ED/src/json_to_conll.py json_data ../../datasets/figer_gold.json $data_directory/figer.conll
#python Fine-ED/src/json_to_conll.py json_data ../../datasets/1k-WFB-g/fner_dev.json $data_directory/fner_dev.conll
#python Fine-ED/src/json_to_conll.py json_data ../../datasets/1k-WFB-g/fner_test.json $data_directory/fner_test.conll

#echo "Generate local variables required for the model."
#python Fine-ED/src/conll_to_tfrecord.py prepare_local_variables $data_directory/train.conll  $glove_vector_file_path unk $data_directory/ 30 --lowercase

#echo "Converting Train data to TFRecord."
#python Fine-ED/src/conll_to_tfrecord.py conll_data $data_directory/ $data_directory/train.conll

#echo "Converting Test data to TFRecord."
#python Fine-ED/src/conll_to_tfrecord.py conll_data $data_directory/ $data_directory/figer.conll --test_data
#python Fine-ED/src/conll_to_tfrecord.py conll_data $data_directory/ $data_directory/fner_dev.conll --test_data
#python Fine-ED/src/conll_to_tfrecord.py conll_data $data_directory/ $data_directory/fner_test.conll --test_data

# Run train test 5 times
for ((i=1; i<=5; i++)); do
  # Do not emit '_run_' from model ckpt name
  # format: prefix_run_suffix
  model_ckpt_name=100_0.5_0.001_500_50_1_run_$i
  
#  echo "Training a LCRF model."
#  time python Fine-ED/src/main_crf.py $data_directory/ $ckpt_directory/$model_ckpt_name 'labels' 'train.conll_*.tfrecord' $rnn_hidden_neurons $keep_prob $learning_rate $batch_size $char_embedding_size $lstm_layers $epochs $save_checkpoint_after --use_char_cnn

#  echo "Testing Figer data."
#  time python Fine-ED/src/main_crf_test_only.py $ckpt_directory/$model_ckpt_name/ $data_directory/figer.conll_0.tfrecord

#  echo "Testing Fner dev data."
#  time python Fine-ED/src/main_crf_test_only.py $ckpt_directory/$model_ckpt_name/ $data_directory/fner_dev.conll_0.tfrecord

#  echo "Testing Fner test data."
#  time python Fine-ED/src/main_crf_test_only.py $ckpt_directory/$model_ckpt_name/ $data_directory/fner_test.conll_0.tfrecord

#  echo "Results Figer data."
#  bash Fine-ED/scripts/report_result.bash Fine-ED/scripts/conlleval.txt $ckpt_directory/$model_ckpt_name/figer.conll_0.tfrecord/ > $ckpt_directory/$model_ckpt_name/figer.conll_0.tfrecord/final_result.txt

#  echo "Results Fner dev data."
#  bash Fine-ED/scripts/report_result.bash Fine-ED/scripts/conlleval.txt $ckpt_directory/$model_ckpt_name/fner_dev.conll_0.tfrecord/ > $ckpt_directory/$model_ckpt_name/fner_dev.conll_0.tfrecord/final_result.txt

#  echo "Results Fner test data."
#  bash Fine-ED/scripts/report_result.bash Fine-ED/scripts/conlleval.txt $ckpt_directory/$model_ckpt_name/fner_test.conll_0.tfrecord/ > $ckpt_directory/$model_ckpt_name/fner_test.conll_0.tfrecord/final_result.txt
done
