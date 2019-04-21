"""
Report final fnet result as per the format needed for paper.
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
    stricts, macros, micros = [], [], []
    with open(filepath) as file_p:
        rows = list(filter(None, file_p.read().split('\n')))
        for i in range(0, len(rows), 7):
            data = rows[i:i+7]
            if len(data) != 7:
                break
            if 'Model no.' in data[0]:
                break
            model_versions.append(int(data[0]))
            # Strict score F1
            stricts.append(float(data[2].split(' ')[-1]))
            # Loose macro score F1
            macros.append(float(data[4].split(' ')[-1]))
            # Loose micro score F1
            micros.append(float(data[6].split(' ')[-1]))
            # If version option is specified return when result is found
            if version and version == model_versions[-1]:
                return stricts[-1], macros[-1], micros[-1]

    assert len(micros) == len(model_versions),\
        print("Error processing ", filepath, "\nLength of micros and models_should match.")

    max_index = np.argmax(micros)
    return {
        'micro' : micros[max_index],
        'macro' : macros[max_index],
        'strict' : stricts[max_index],
        'model_version' : model_versions[max_index]
    }

#pylint:disable=invalid-name
if __name__ == '__main__':
    master_ckpt_directory = sys.argv[1]
    directories = glob.glob(master_ckpt_directory + '/*/')

    config_ckpt_mapping = {}

    for directory in directories:
        if '_run_' not in directory:
            continue
        ckpt_name = os.path.basename(directory[:-1])
        prefix = ckpt_name.split('_run_')[0]
        if prefix not in config_ckpt_mapping:
            config_ckpt_mapping[prefix] = []
        config_ckpt_mapping[prefix].append(ckpt_name)


    datasets_to_process = ['figer_gold.json_0.tfrecord', 'fner_dev.json_0.tfrecord',
                           'figer_gold_lstm_crf.json_0.tfrecord',
                           'fner_dev_lstm_crf.json_0.tfrecord'
                           ]

    # Development evaluation mapping.
    dev_eval_map = {
        'fner_dev.json_0.tfrecord' : 'fner_test.json_0.tfrecord',
        'fner_dev_lstm_crf.json_0.tfrecord' : 'fner_test_lstm_crf.json_0.tfrecord',
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
                        'stricts' : [],
                        'macros' : [],
                        'micros' : [],
                        'model_versions' : [],
                        'ckpt_names' : []
                    }
                results[config][dataset]['stricts'].append(result['strict'])
                results[config][dataset]['macros'].append(result['macro'])
                results[config][dataset]['micros'].append(result['micro'])
                results[config][dataset]['model_versions'].append(result['model_version'])
                results[config][dataset]['ckpt_names'].append(ckpt_name)

    for config in results:
        for dataset in results[config]:
            print()
            print("Result for ", config, "dataset: ", dataset)
            result = results[config][dataset]
            arg_max = np.argmax(result['micros'])
            print(str(result['stricts'][arg_max]) + '\t'
                  + str(result['macros'][arg_max]) + '\t'
                  + str(result['micros'][arg_max]))
            print(str(round(np.std(result['stricts']), 2)) + '\t'
                  + str(round(np.std(result['macros']), 2)) + '\t'
                  + str(round(np.std(result['micros']), 2)))
            print('Best model: ', result['ckpt_names'][arg_max],
                  'Version: ', result['model_versions'][arg_max])

            # Eval set result
            if dataset in dev_eval_map:
                s_eval, ma_eval, mi_eval = parse_result_file(
                    master_ckpt_directory +
                    result['ckpt_names'][arg_max] +
                    '/' + dev_eval_map[dataset] + '/final_result.txt',
                    int(result['model_versions'][arg_max])
                )
                print('Eval set result:')
                print(str(s_eval) + '\t' + str(ma_eval) + '\t' + str(mi_eval))
