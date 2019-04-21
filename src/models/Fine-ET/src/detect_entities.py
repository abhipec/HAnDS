"""
Convert conll format output of any named entity detection system with its
Json format data to json format data suitable to implement pipeline based fine
named entity recognition.
"""
import sys
import json


def bio_tag_to_entity_mentions(bio_tags):
    """
    Return start and end indices of entities in BIO format list.
    """
    within_entity_mention = False
    mentions = []
    for i, tag in enumerate(bio_tags):
        if not within_entity_mention:
            # start of entity mention
            if 'B-' in tag:
                start = i
                within_entity_mention = True
        else:
            if tag == 'O':
                end = i
                mentions.append({
                    'start' : start,
                    'end' : end,
                    'labels' : []
                })
                within_entity_mention = False
            # two entities side by side without O tag
            if 'B-' in tag:
                end = i
                mentions.append({
                    'start' : start,
                    'end' : end,
                    'labels' : []
                })
                start = i
    if within_entity_mention:
        end = len(bio_tags)
        mentions.append({
            'start' : start,
            'end' : end,
            'labels' : []
        })
    return mentions

#pylint:disable=invalid-name
if __name__ == '__main__':
    input_json_file = sys.argv[1]
    input_conll_file = sys.argv[2]
    output_filename = sys.argv[3]

    with open(input_json_file, 'r', encoding='utf-8') as file_j,\
            open(input_conll_file, 'r', encoding='utf-8') as file_c,\
            open(output_filename, 'w', encoding='utf-8') as file_o:
        json_lines = filter(None, file_j.read().split('\n'))
        conll_sentences = filter(None, file_c.read().split('\n\n'))
        for json_line, conll_sentence in zip(json_lines, conll_sentences):
            tags = [x.split(' ')[-1] for x in conll_sentence.split('\n')]
            json_data = json.loads(json_line)
            json_data['mentions'] = bio_tag_to_entity_mentions(tags)
            json.dump(json_data, file_o)
            file_o.write('\n')
