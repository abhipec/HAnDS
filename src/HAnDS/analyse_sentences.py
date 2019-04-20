"""
Produce different information about the dataset generated.
Create a pickle object for label distribution information to be used during sentence sampling.
The information include:
Mean, median sentence length.
Total sentences.
Total tokens.
Unique tokens.
Total mentions.
Unique entities.
Unique mentions.
Average label per entity.
Label distribution: text file.
Usage:
    analyse_sentences complete_data <sentence_directory> [--label_distribution_file=<filepath>]
                                    [--pickle_path=<filepath>] [--generate_label_reference]
    analyse_sentences single_file <filepath> [--label_distribution_file=<filepath>]
    analyse_sentences (-h | --help)

Options:
    -h, --help                          Display help message.
    <sentence_directory>                Output directory of correct sentences.
    <label_distribution_file>           File path where label distribution will be stored.
    <pickle_path>                       File path where pickle dump of distribution will be written.
    <filepath>                          File path of json file.

"""
import sys
import glob
import gzip
import json
import pickle
import numpy as np
from docopt import docopt

__author__ = "Abhishek"
__maintainer__ = "Abhishek"

#pylint:disable=too-many-locals,too-many-branches,too-many-statements
def process_files(filepaths, label_distribution_file=None,
                  pickle_path=None, generate_label_reference=False):
    """
    Process list of files and produce different statistics.
    """
    label_distribution = {}
    total_entities = 0
    total_tokens = 0
    entities = set()
    surface_names = set()
    tokens = set()
    length_distribution = []
    label_reference = {}
    label_reference[0] = set()

    for file in filepaths:
        print(file)
        use_gzip = True if file[-3:] == '.gz' else False
        if use_gzip:
            file_p = gzip.GzipFile(file, 'r')
        else:
            file_p = open(file, encoding='utf-8')

        for row in file_p:
            if use_gzip:
                json_data = json.loads(row.decode('utf-8'))
            else:
                json_data = json.loads(row)
            try:
                did = json_data['did']
            except KeyError:
                did = json_data['fileid']
            pid = json_data.get('pid', 0)
            try:
                sid = json_data['sid']
            except KeyError:
                sid = json_data['senid']
            uid = '_'.join([str(did), str(pid), str(sid)])

            length = len(json_data['tokens'])
            length_distribution.append(length)

            # Sentences with no entity mentions
            if 'mentions' in json_data:
                mentions = json_data['mentions']
            elif 'links' in json_data:
                mentions = json_data['links']
            else:
                print('ERROR')
                sys.exit(1)
            mention_count = len(mentions)
            if mention_count == 0 and generate_label_reference:
                label_reference[0].add(uid)

            # Token stats
            for token in json_data['tokens']:
                tokens.add(token)
                total_tokens += 1

            # Mention stats
            for mention in mentions:
                total_entities += 1
                entities.add(mention.get('link', 0))
                surface_names.add(mention.get('name', 0))
                # Label stats
                for label in mention['labels']:
                    if label not in label_distribution:
                        label_distribution[label] = 0
                    label_distribution[label] += 1

                    if generate_label_reference:
                        if label not in label_reference:
                            label_reference[label] = set()
                        label_reference[label].add(uid)

    label_count = sum(label_distribution.values())

    if label_distribution_file:
        with open(label_distribution_file, 'w', encoding='utf-8') as file_p:
            for label in sorted(label_distribution, key=label_distribution.get):
                file_p.write('| ' + label + ' | ' +
                             str(label_distribution[label]) + ' | ' +
                             str(round((label_distribution[label]/label_count)*100, 2)) + ' |\n')
    if pickle_path:
        pickle.dump((label_distribution, label_reference),
                    open(pickle_path, 'wb'))

    # Unique labels
    print(len(label_distribution))
    # Total entities
    print(total_entities)
    # Unique entities
    print(len(entities))
    # Unique mentions
    print(len(surface_names))
    # Total sentences
    print(len(length_distribution))
    # Total tokens
    print(total_tokens)
    # Unique tokens
    print(len(tokens))
    # Average labels per entity mention
    print(label_count/total_entities)
    # Average sentence length
    print(np.mean(length_distribution))
    # Median sentence length
    print(np.median(length_distribution))
    # Cumulative distribution function of labels
    print()
    for label in label_distribution:
        label_distribution[label] = (100 * label_distribution[label]) / label_count

    c_sum = 0
    for label in sorted(label_distribution, key=label_distribution.get, reverse=True):
        print(label_distribution[label] + c_sum)
        c_sum += label_distribution[label]

#pylint:disable=invalid-name
if __name__ == '__main__':
    cmd_arguments = docopt(__doc__)
    if cmd_arguments['complete_data']:
        process_files(list(glob.iglob(cmd_arguments['<sentence_directory>'] + '**/wiki_*')),
                      cmd_arguments['--label_distribution_file'],
                      cmd_arguments['--pickle_path'],
                      cmd_arguments['--generate_label_reference'])
    if cmd_arguments['single_file']:
        process_files([cmd_arguments['<filepath>']], cmd_arguments['--label_distribution_file'])
