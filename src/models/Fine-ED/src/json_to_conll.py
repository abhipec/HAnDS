"""
Convert the .json data format to CoNLL format for entity detection.
Usage:
    json_to_conll json_data <input_file_path> <output_file_path>
    json_to_conll (-h | --help)

Options:
    -h, --help                              Display help message.
"""
import json
import gzip
from docopt import docopt

def remove_overlapping_ems(mentions):
    """
    Spotlight can generate overlapping EMs.
    Among the intersecting EMs, remove the smaller ones.
    """
    to_remove = set()
    new_mentions = []
    length = len(mentions)
    for i in range(length):
        start_r = mentions[i]['start']
        end_r = mentions[i]['end']
        for j in range(length):
            if i != j and j not in to_remove:
                start = mentions[j]['start']
                end = mentions[j]['end']
                if start_r >= start and end_r <= end:
                    to_remove.add(i)
    for i in range(length):
        if i not in to_remove:
            new_mentions.append(mentions[i])
    return new_mentions

def generate_bio_entity_tags(mentions, sentence_length):
    """
    Generate BIO/IOB2 tags for entity detection task.
    """
    bio_tags = ['O'] * sentence_length
    for mention in mentions:
        start = mention['start']
        end = mention['end']
        bio_tags[start] = 'B-E'
        for i in range(start + 1, end):
            bio_tags[i] = 'I-E'
    return bio_tags

def convert_json_data(input_file_path, output_file_path):
    "Convert json data format to fned conll format."
    use_gzip = True if input_file_path[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(input_file_path, 'r')
    else:
        file_i = open(input_file_path, encoding='utf-8')
    with open(output_file_path, 'w', encoding='utf-8') as file_o:
        for row in file_i:
            if use_gzip:
                json_data = json.loads(row.decode('utf-8'))
            else:
                json_data = json.loads(row)
            tokens = json_data['tokens']
            try:
                entity_mentions = json_data['mentions']
            except KeyError:
                entity_mentions = json_data['links']

            filtered_ems = remove_overlapping_ems(entity_mentions)
            tags = generate_bio_entity_tags(filtered_ems, len(tokens))
            for token, tag in zip(tokens, tags):
                file_o.write(token + ' ' + tag + '\n')
            file_o.write('\n')
    file_i.close()

#pylint:disable=invalid-name
if __name__ == '__main__':
    cmd_arguments = docopt(__doc__)
    if cmd_arguments['json_data']:
        print('Converting json data to fned conll format.')
        convert_json_data(cmd_arguments['<input_file_path>'], cmd_arguments['<output_file_path>'])
