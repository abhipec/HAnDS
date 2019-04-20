"""
Apply stage-I of HAnDS framework.
"""

import sys
import os
import json
import glob
import gzip
from helper import prepare_dictionaries, ensure_directory, is_surface_name_referential
from helper import is_title_entity, get_actual_title


def parse_single_file(input_file_path, output_file_path, dictionaries):
    """
    Apply HAnDS stage-I on single file.
    """
    with gzip.GzipFile(input_file_path, 'r') as file_i,\
            gzip.GzipFile(output_file_path, 'w') as file_o:
        for row in file_i:
            json_data = json.loads(row.decode('utf-8'))
            new_paragraphs = []
            for paragraph in json_data['paragraphs']:
                text = paragraph[0]
                links = paragraph[1]
                sentences = paragraph[2]
                new_links = []
                for link in links:
                    # Check is it is an entity, non-entity or a undecided link
                    # Discard undecided link
                    if is_title_entity(link['link'], dictionaries) == -1:
                        continue
                    if not is_surface_name_referential(link['name'], link['link'], dictionaries):
                        continue
                    # Ensure uniformity across links
                    link['link'] = get_actual_title(link['link'], dictionaries)
                    new_links.append(link)
                new_paragraphs.append([
                    text, new_links, sentences
                ])
            json_data['paragraphs'] = new_paragraphs
            json_str = json.dumps(json_data) + '\n'
            file_o.write(json_str.encode('utf-8'))

#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Check README.md for usage details.")
        sys.exit(1)
    master_dictionary_filepath = sys.argv[1]
    label_mapping_filepath = sys.argv[2]
    input_directory = sys.argv[3]
    output_directory = sys.argv[4]

    all_dictionaries = prepare_dictionaries(master_dictionary_filepath, label_mapping_filepath)

    filepaths = list(glob.iglob(input_directory + '**/wiki_*'))
    for filepath in filepaths:
        print(filepath)
        subdir = os.path.basename(os.path.split(filepath)[0])
        basename = os.path.split(filepath)[1]
        ensure_directory(output_directory + subdir)
        output_path = output_directory + subdir + '/' + basename
        if os.path.isfile(output_path):
            continue
        parse_single_file(filepath, output_path, all_dictionaries)
