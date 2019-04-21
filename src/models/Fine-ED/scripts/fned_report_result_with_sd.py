"""
Report final fned result as per the format needed for paper.
"""

import sys
import os
import glob
import numpy as np

def parse_result_file(filepath, version=None):
    """
    Parse single fned result file.
    """
    model_versions = []
    precisions, recalls, fb1s = [], [], []
    with open(filepath) as file_p:
        for row in filter(None, file_p.read().split('\n')):
            if 'Maximum F1 score:' in row:
                break
            if 'processed' in row:
                continue
            if row.isdecimal() and '.' not in row:
                model_versions.append(int(row))
                continue
            if 'accuracy:' in row:
                _, precision, recall, fb1 = [x.split(':')[-1].strip() for x in row.split(';')]
                precisions.append(float(precision[:-1]))
                recalls.append(float(recall[:-1]))
                fb1s.append(float(fb1))
                # If version option is specified return when result is found
                if version and version == model_versions[-1]:
                    return precisions[-1], recalls[-1], fb1s[-1]
    assert len(precisions) == len(model_versions),\
        print("Error processing ", filepath, "\nLength of precisions and models_should match.")

    max_index = np.argmax(fb1s)
    return {
        'precision' : precisions[max_index],
        'recall' : recalls[max_index],
        'fb1' : fb1s[max_index],
        'model_version' : model_versions[max_index]
    }

#pylint:disable=invalid-name
if __name__ == '__main__':
    master_ckpt_directory = sys.argv[1]
    directories = glob.glob(master_ckpt_directory + '/*/')

    train_dataset_name = os.path.basename(os.path.split(master_ckpt_directory[:-1])[0])
    copy_dir = '../../results/Fine-ED/lstm_crf/' + train_dataset_name + '/'
    config_ckpt_mapping = {}

    for directory in directories:
        ckpt_name = os.path.basename(directory[:-1])
        prefix = ckpt_name.split('_run_')[0]
        if prefix not in config_ckpt_mapping:
            config_ckpt_mapping[prefix] = []
        config_ckpt_mapping[prefix].append(ckpt_name)

    datasets_to_process = ['figer.conll_0.tfrecord', 'fner_dev.conll_0.tfrecord']

    # Development evaluation mapping.
    dev_eval_map = {
        'fner_dev.conll_0.tfrecord' : 'fner_test.conll_0.tfrecord'
    }

    results = {}
    for config in config_ckpt_mapping:
        for ckpt_name in config_ckpt_mapping[config]:
            for dataset in datasets_to_process:
                result_dir = master_ckpt_directory + ckpt_name + '/' + dataset + '/'
                result = parse_result_file(result_dir + 'final_result.txt')
                if config not in results:
                    results[config] = {}
                if dataset not in results[config]:
                    results[config][dataset] = {
                        'precisions' : [],
                        'recalls' : [],
                        'fb1s' : [],
                        'model_versions' : [],
                        'ckpt_names' : []
                    }
                results[config][dataset]['precisions'].append(result['precision'])
                results[config][dataset]['recalls'].append(result['recall'])
                results[config][dataset]['fb1s'].append(result['fb1'])
                results[config][dataset]['model_versions'].append(result['model_version'])
                results[config][dataset]['ckpt_names'].append(ckpt_name)
    to_print = []
    to_print.append('mkdir -p ' + copy_dir)
    for config in results:
        for dataset in results[config]:
            print("Testing dataset:\t\t", dataset)
            print("Best training directory:\t", config)
            result = results[config][dataset]
            arg_max = np.argmax(result['fb1s'])
            print("Development set results (P, R, F1):")
            print(str(result['precisions'][arg_max]) + '\t'
                  + str(result['recalls'][arg_max]) + '\t'
                  + str(result['fb1s'][arg_max]))
            print("Development set standard deviation (P, R, F1):")
            print(str(round(np.std(result['precisions']), 2)) + '\t'
                  + str(round(np.std(result['recalls']), 2)) + '\t'
                  + str(round(np.std(result['fb1s']), 2)))
            # Helping commands to copy best result
            dest_file_name = dataset[:dataset.find('.')] + '.conll'
            to_print.append('cp ' + master_ckpt_directory + result['ckpt_names'][arg_max]
                            + '/' + dataset + '/' + str(result['model_versions'][arg_max])
                            + ' ' + copy_dir + dest_file_name)
            # Eval set result
            if dataset in dev_eval_map:
                p_eval, r_eval, f_eval = parse_result_file(
                    master_ckpt_directory +
                    result['ckpt_names'][arg_max] +
                    '/' + dev_eval_map[dataset] + '/final_result.txt',
                    int(result['model_versions'][arg_max])
                )
                print('Test set results:')
                print(str(p_eval) + '\t' + str(r_eval) + '\t' + str(f_eval))
                dest_file_name = dev_eval_map[dataset][:dev_eval_map[dataset].find('.')] + '.conll'
                to_print.append('cp ' + master_ckpt_directory + result['ckpt_names'][arg_max]
                                + '/' + dev_eval_map[dataset]
                                + '/' + str(result['model_versions'][arg_max])
                                + ' ' + copy_dir + dest_file_name)
    print("")
    instruction_str = """To use the Fine-ED results as a input to Fine-ET system,\
 then execute the following commands. These commands will save the best results as .conll file,\
 which will be then later used by the Fine-ET model."""
    print(instruction_str)
    for print_str in to_print:
        print(print_str)
