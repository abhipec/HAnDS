"""
Run StanfordNERTagger on json format data and dump the final result in CoNLL format.
"""
import sys
import json
from nltk.tag import StanfordNERTagger

#pylint:disable=invalid-name
def output_to_IOB2_string(output):
    """
    Convert Stanford NER tags to IOB2 tags.
    """
    iob2_tags = []
    names = []
    previous_tag = 'O'
    for _, tup in enumerate(output):
        name, tag = tup
        if tag != 'O':
            tag = 'E'

        if tag == 'O':
            iob2_tags.append(tag)
            previous_tag = tag
            names.append(name)
        else:
            if previous_tag == 'O':
                iob2_tags.append('B-' + tag)
            else:
                iob2_tags.append('I-' + tag)
            previous_tag = tag
            names.append(name)
    return names, iob2_tags

def mentions_to_IOB2(mention_list, sentence_length):
    """
    Convert json mentions to IOB2 tags.
    """
    tags = ['O'] * sentence_length
    for mention in mention_list:
        tags[mention['start']] = 'B-E'
        for i in range(mention['start'] + 1, mention['end']):
            tags[i] = 'I-E'
    return tags

#pylint:disable=invalid-name
if __name__ == '__main__':
    stanford_ner_directory = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]
    nlp_stanford = StanfordNERTagger(
        stanford_ner_directory + 'classifiers/english.conll.4class.distsim.crf.ser.gz',
        stanford_ner_directory + 'stanford-ner-3.7.0.jar',
        encoding='utf-8'
    )
    annotations_stanford = []
    with open(input_filename) as file_p,\
            open(output_filename, 'w') as file_o:
        for row in file_p:
            json_data = json.loads(row)
            tokens = json_data['tokens']
            gold_tags = mentions_to_IOB2(json_data['mentions'], len(tokens))
            names_list, predicted_tags = output_to_IOB2_string(nlp_stanford.tag(tokens))
            for name_value, gold_tag, predicted_tag in zip(names_list, gold_tags, predicted_tags):
                file_o.write(name_value + ' ' + gold_tag + ' ' + predicted_tag + '\n')
            file_o.write('\n')
