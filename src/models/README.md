# Fine-ED and Fine-ET models

## Fine Entity Detection model

To train the learning model refer to the Fine-ED/scripts directory. Refer to any bash script such as RS_fner_2.bash, all necessary steps are included in the script. The name of the script denotes which dataset was used for training. The script can be run using the following command. By default, all commands in the script are commented out.

```
bash Fine-ED/scripts/RS_fner_2.bash
```


### Report results

The following code will report the max result and standard deviation of results obtained by multiple trainings. 

```
# Usage python Fine-ED/scripts/fned_report_result_with_sd.py path_to_checkpoints_directory/
python Fine-ED/scripts/fned_report_result_with_sd.py ../../model_training_data/Fine-ED/RS_fner/ckpts/
```
The above command will also give instruction on using the best output of Fine-ED in pipelined manner to Fine-ET model.

## Fine Entity Typing model

Compile the necessary C files. It should work with gcc 5 and above.

```
cd Fine-ET/src/lib
bash compile_gcc_5.bash
```

To train the learning model refer to the Fine-ET/src/scripts directory. Refer to any bash script such as RS_fner_2.bash, all necessary steps are included in the script. The name of the script denotes which dataset was used for training. The script can be run using the following command. By default, all commands in the script are commented out.

```
bash Fine-ET/src/scripts/RS_fner_2.bash
```


### Report results

The following code will report the max result and standard deviation of results obtained by multiple trainings. 

```
# Usage python Fine-ET/src/scripts/fnet_report_result_with_sd.py path_to_checkpoints_directory/
python Fine-ET/src/scripts/fnet_report_result_with_sd.py ../../model_training_data/Fine-ET/RS_fner/ckpts/
```

## Tagger model of https://github.com/glample/tagger

```
bash tagger/tagger_figer.bash
bash tagger/tagger_fner_dev.bash
bash tagger/tagger_fner_test.bash
```
Please check the corresponding files for more details.
