"""
Apply stage-II of HAnDS framework.
"""
import sys
import os
import glob
import gzip
import json
import pygtrie
from helper import ensure_directory, prepare_dictionaries, separate_links
from helper import is_entity_lowercase_dominant, get_actual_title, generate_candidate_names

def tree_of_candidates(link_set, dictionaries):
    """
    Construct a trie tree for entities and non_entities.
    """
    tree = pygtrie.CharTrie()
    if not link_set:
        return tree
    to_remove = set()
    for link in link_set:
        title = get_actual_title(link, dictionaries)
        for name in generate_candidate_names(title, dictionaries):
            if name in tree:
                to_remove.add(name)
            tree[name] = title
    # Remove names that could refer to two or more links.
    for name in to_remove:
        del tree[name]
    return tree

def find_capital_character(string, offset):
    """
    Find capital character in a string after offset.
    """
    for index in range(offset, len(string)):
        if string[index].isupper():
            return index
    return None

def find_character(string, offset):
    """
    Find character in a string after offset.
    """
    for index in range(offset, len(string)):
        if string[index].isalpha():
            return index
    return None

def generate_unlinked_strings(para_string, link_offsets):
    """
    Produce strings that do not overlap with any infolink.
    """
    padded_offsets = [{'end': 0}] + link_offsets + [{'start': len(para_string)}]
    for start_o, end_o in zip(padded_offsets, padded_offsets[1:]):
        yield para_string[start_o['end']:end_o['start']], start_o['end']

def annotate_unlinked_string(string, tree, string_para_offset):
    """
    Annotate a string with possible entities.
    """
    start_index = find_capital_character(string, 0)
    new_link_offsets = []
    while start_index is not None:
        string_to_match = string[start_index:]
        (k, value) = tree.longest_prefix(string_to_match)
        # do not allow length one string match
        if k and len(k) == 1:
            k = None
        if k is None:
            start_index = find_capital_character(string, start_index + 1)
        else:
            new_link_offsets.append({
                'start' : string_para_offset + start_index,
                'end' : string_para_offset + start_index + len(k),
                'name' : k,
                'link' : value
            })
            start_index = find_capital_character(string, start_index + len(k))
    return new_link_offsets

def annotate_unlinked_lowercase_string(string, tree, string_para_offset):#pylint:disable=invalid-name
    """
    Annotate a string with possible entities.
    """
    start_index = find_character(string, 0)
    new_link_offsets = []
    while start_index is not None:
        string_to_match = string[start_index:]
        (k, value) = tree.longest_prefix(string_to_match)
        # do not allow length one string match
        if k and len(k) == 1:
            k = None
        if k is None:
            start_index = find_character(string, start_index + 1)
        else:
            new_link_offsets.append({
                'start' : string_para_offset + start_index,
                'end' : string_para_offset + start_index + len(k),
                'name' : k,
                'link' : value
            })
            start_index = find_character(string, start_index + len(k))
    return new_link_offsets

def process_paragraph(para_string, offsets, tree_tuple):
    """
    Annotate a single paragraph.
    """
    tree_all, tree_entities = tree_tuple

    title_case_links = []
    lowercase_links = []
    sorted_links = sorted(offsets, key=lambda x: x['start'])
    for string, string_para_offset in generate_unlinked_strings(para_string, sorted_links):
        title_case_links += annotate_unlinked_string(string, tree_all, string_para_offset)

    new_sorted_links = sorted(sorted_links + title_case_links, key=lambda x: x['start'])
    for string, string_para_offset in generate_unlinked_strings(para_string,
                                                                new_sorted_links):
        lowercase_links += annotate_unlinked_lowercase_string(string,
                                                              tree_entities,
                                                              string_para_offset)

    new_sorted_links = sorted(sorted_links + title_case_links + lowercase_links,
                              key=lambda x: x['start'])
    return new_sorted_links

def parse_single_file(input_file_path, output_file_path, dictionaries):#pylint:disable=too-many-locals
    """
    Apply HAnDS stage-II on single file.
    """
    #pylint:enable=too-many-locals
    with gzip.GzipFile(input_file_path, 'r') as file_i,\
            gzip.GzipFile(output_file_path, 'w') as file_o:
        for row in file_i:
            json_data = json.loads(row.decode('utf-8'))
            new_paragraphs = []
            # Partition all outgoing links of a document into an entity, non-entity and undecided
            # link.
            entities_tuple = separate_links(json_data['outgoing_links'], dictionaries)
            entities, non_entities, _ = entities_tuple
            lowercase_dominant_entities = set([x for x in entities
                                               if is_entity_lowercase_dominant(x, dictionaries)])
            tree_all = tree_of_candidates(entities.union(non_entities), dictionaries)
            tree_lowercase_entities = tree_of_candidates(lowercase_dominant_entities, dictionaries)

            for paragraph in json_data['paragraphs']:
                text = paragraph[0]
                links = paragraph[1]
                sentences = paragraph[2]
                new_links = process_paragraph(text, links, (tree_all, tree_lowercase_entities))
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
