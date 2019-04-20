"""
In this step, remove non-entities and store entity lables in Json format.
"""

import sys
import os
import glob
import gzip
import json
from helper import ensure_directory, prepare_dictionaries, return_mapped_labels
from helper import return_parent_label, get_actual_title


def return_test_sentences(test_datafile):
    """
    Return sentences present in test file as a set.
    """
    sentences = set()
    with open(test_datafile, encoding='utf-8') as file_p:
        for row in file_p:
            json_data = json.loads(row)
            tokens = json_data['tokens']
            sentences.add(' '.join(tokens))
    # Add the following sentences manually due to some pre-processing differences.
    # Verified on the processed Wikipedia dump provided with the code.
    sentences.add('Georges - François - Xavier - Marie Grente ( 5 May 1872 -- 5 May 1959 ) was a French Cardinal of the Roman Catholic Church .')
    sentences.add('The Ferrari 458 Italia is a mid - engined sports car produced by the Italian sports car manufacturer Ferrari .')
    sentences.add('In addition the RAF regiment was formed in 1941 with responsibility for airfield air defence , eventually with Bofors 40 mm as their main armament .')
    sentences.add('For his contribution to the television industry , George Putnam was awarded three Emmy awards as well as a star on the Hollywood Walk of Fame at 6372 Hollywood Blvd. The late Ted Knight stated that Putnam served in part as a model for the Ted Baxter character in the 1970s television series " The Mary Tyler Moore Show " on CBS .')
    sentences.add("The railroad 's ancient 2 -6-0 steam locomotive was soon replaced with a new GE 44 - ton switcher diesel locomotive .")
    sentences.add('Health 2.0 built on the possibilities for changing health care , which started with the introduction of eHealth in the mid - 1990s following the emergence of the World Wide Web .')
    sentences.add('In 1927 , the Dodge Fast Four was the new mid - level car from Dodge .')
    return sentences

def parse_single_file(input_file_path, output_file_path, dictionaries, sentences, count_dict):
    """
    Add labels from KB, to entity mentions. Remove non-entity mentions.
    """
    with gzip.GzipFile(input_file_path, 'r') as file_i,\
            gzip.GzipFile(output_file_path, 'w') as file_o:
        for row in file_i:
            json_data = json.loads(row.decode('utf-8'))
            tokens = json_data['tokens']
            sentence = ' '.join(tokens)
            if sentence in sentences:
                count_dict[sentence] += 1
                print(sentence)
                continue
            new_links = []
            for link in json_data['links']:
                labels = return_mapped_labels(link['link'], dictionaries)
                if labels:
                    for label in list(labels):
                        labels.add(return_parent_label(label))
                    link['labels'] = list(labels)
                    link['link'] = get_actual_title(link['link'], all_dictionaries)
                    new_links.append(link)
            json_data['links'] = new_links
            json_str = json.dumps(json_data) + '\n'
            file_o.write(json_str.encode('utf-8'))



#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Check README.md for usage details.")
        sys.exit(1)
    master_dictionary_filepath = sys.argv[1]
    label_mapping_filepath = sys.argv[2]
    test_data_file = sys.argv[3]
    input_directory = sys.argv[4]
    output_directory = sys.argv[5]

    all_dictionaries = prepare_dictionaries(master_dictionary_filepath, label_mapping_filepath)

    test_sentences = return_test_sentences(test_data_file)
    count_dict = {}
    for sentence in test_sentences:
        count_dict[sentence] = 0
    filepaths = list(glob.iglob(input_directory + '**/wiki_*'))
    for filepath in filepaths:
        print(filepath)
        subdir = os.path.basename(os.path.split(filepath)[0])
        basename = os.path.split(filepath)[1]
        ensure_directory(output_directory + subdir)
        output_path = output_directory + subdir + '/' + basename
        if os.path.isfile(output_path):
            continue
        parse_single_file(filepath, output_path, all_dictionaries, test_sentences, count_dict)
    for sentence in sorted(count_dict, key=count_dict.get):
        print(count_dict[sentence], '\t', sentence)
