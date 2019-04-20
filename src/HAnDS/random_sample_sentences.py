"""
Randomly sample sentences from the corpus generated using the HAnDS framework.
"""


import sys
import pickle
import glob
import json
import random
import gzip

def sample_sentences(input_directory,#pylint:disable=too-many-locals
                     label_analysis_pickle_file,
                     output_file_path,
                     number_of_samples):
    """
    Sample a desired number of sentences from the generated datasets.
    """
    _, label_reference_dict = pickle.load(open(label_analysis_pickle_file, 'rb'))
    all_sentences = set()
    for key in label_reference_dict:
        all_sentences.update(label_reference_dict[key])

    sampled_sentences = set(random.sample(all_sentences, number_of_samples))

    with gzip.GzipFile(output_file_path, 'w') as file_o:
        for file in glob.iglob(input_directory + '**/wiki_*'):
            print(file)
            with gzip.GzipFile(file, 'r') as file_p:
                for row in file_p:
                    json_data = json.loads(row.decode('utf-8'))
                    uid = '_'.join([
                        str(json_data['did']),
                        str(json_data['pid']),
                        str(json_data['sid'])])
                    if uid in sampled_sentences:
                        json_str = json.dumps(json_data) + '\n'
                        file_o.write(json_str.encode('utf-8'))


if __name__ == '__main__':
    sample_sentences(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
