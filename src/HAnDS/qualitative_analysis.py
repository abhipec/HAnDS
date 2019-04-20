"""
Qualitative analysis of dataset generated using CADS vs NDS approach.
"""

import sys
import glob
import json
import gzip

def sentence_id(json_sentence):
    """
    Return the unique if of a sentence.
    """
    return '_'.join([
        str(json_sentence['did']),
        str(json_sentence['pid']),
        str(json_sentence['sid'])
    ])


def parse_dataset(input_directory, restrict=None):
    """
    Parse the wikipedia annotated json files.
    """
    filepaths = sorted(glob.iglob(input_directory + '**/wiki_*'))
    entity_mentions = set()
    sentences = set()
    unique_entities = set()
    for filepath in filepaths:
        print(filepath)
        with gzip.GzipFile(filepath, 'r') as file_i:
            for row in file_i:
                json_data = json.loads(row.decode('utf-8'))
                s_id = sentence_id(json_data)
                if not restrict:
                    for link in json_data['links']:
                        start = link['start']
                        end = link['end']
                        e_ids = str(start) + '_' + str(end) + '_' + s_id
                        entity_mentions.add(e_ids)
                        unique_entities.add(link['link'])
                    sentences.add(s_id)
                else:
                    if s_id in restrict:
                        for link in json_data['links']:
                            start = link['start']
                            end = link['end']
                            e_ids = str(start) + '_' + str(end) + '_' + s_id
                            entity_mentions.add(e_ids)
                            unique_entities.add(link['link'])
                        sentences.add(s_id)
    return entity_mentions, sentences, unique_entities

#pylint:disable=invalid-name
if __name__ == '__main__':
    ems_hands, sentences_hands, ue_hands = parse_dataset(sys.argv[1])
    ems_nds, sentences_nds, ue_nds = parse_dataset(sys.argv[2], sentences_hands)
    print("Entity mention analysis")
    print('Total entity mentions HAnDS:', len(ems_hands))
    print('Total entity mentions NDS:', len(ems_nds))
    print('|H - N|', len(ems_hands - ems_nds))
    print('|H intersection N|', len(ems_hands.intersection(ems_nds)))
    print('|N - H|', len(ems_nds - ems_hands))

    print("Entity analysis")
    print('Total entity HAnDS:', len(ue_hands))
    print('Total entity NDS:', len(ue_nds))
    print('|H - N|', len(ue_hands - ue_nds))
    print('|H intersection N|', len(ue_hands.intersection(ue_nds)))
    print('|N - H|', len(ue_nds - ue_hands))
