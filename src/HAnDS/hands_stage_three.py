# coding: latin-1
"""
Apply stage-III of HAnDS framework.
"""
import os
import sys
import glob
import gzip
import json
from helper import ensure_directory

__author__ = "Abhishek, Sanya B. Taneja, and Garima Malik"
__maintainer__ = "Abhishek"

def read_frequent_sentence_starters(file_path, top_x):
    """
    Load top x number of lines from file.
    """
    words = set()
    count = 0
    with open(file_path, encoding='utf-8') as file_r:
        for line in file_r:
            word = line.split(' ')[0]
            words.add(word)
            count += 1
            if count == top_x:
                break
    return words

def check_segmentation_error(json_sentence):
    """
    Return True if there is a sentence segmentation error.
    """
    valid = True
    tokens = json_sentence['tokens']
    first_token_invalid_set = set(['.', '...', ',', '[', '--', '-', '/',
                                   ';', ':', '*', '•', "'s'"])
    if len(tokens) <= 5:
        valid = False
    if tokens[0].isalpha() and tokens[0].islower() and all(ord(char) < 128 for char in tokens[0]):
        valid = False
    if tokens[0] in first_token_invalid_set:
        valid = False
    string = ' '.join(tokens)
    lrb_count = string.count('(')
    rrb_count = string.count(')')
    if lrb_count == 1 and rrb_count == 0:
        valid = False
    if lrb_count == 0 and rrb_count == 1:
        valid = False
    quotes = string.count('"')
    degree = string.count('°')
    if quotes == 1 and degree == 0:
        valid = False
    return not valid

def is_sentence_valid(json_sentence, frequent_words):
    """
    Return True is sentence is valid.
    """
    if check_segmentation_error(json_sentence):
        return False

    titles = set(['Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Hon.', 'Prof.', 'Miss', 'Rev.', 'St.',
                  'Pres.', 'Supt.', 'Rep.', 'Sen.', 'Gov.', 'Amb.', 'Treas.', 'Sec.',
                  'Pvt.', 'Spec.', 'Sgt.', 'Ens.', 'Adm.', 'Maj.', 'Capt.', 'Lt.', 'Col.',
                  'Gen.', 'Junior', 'Senior', 'Professor', 'President', 'Minister',
                  'Senator', 'Governor', 'Secretary', 'General', 'Lieutenant', 'Colonel',
                  'Sergeant', 'Major', 'Captain', 'Admiral', 'Reverend', 'Ambassador',
                  'Jr.', 'Sr.'])

    days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    months = set(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                  'September', 'October', 'November', 'December'])

    tokens = json_sentence['tokens']
    poss = json_sentence['pos']
    tags = generate_bio_entity_tags(tokens, json_sentence['links'])
    valid = True

    # First check for sentence segmentation errors.
    # check for POS tag and frequent words only for correctly segmented sentences.
    ss_allowed_postags = set(['DT', 'IN', 'PRP', 'RB', 'JJ', 'VBG', 'PRP$', 'CD', 'EX', 'WRB',
                              'VBN', '``', 'CC', 'TO', 'JJS', 'RBR', 'JJR', 'PDT', 'WDT', 'WP',
                              'LS', 'RBS', 'MD', 'VBP', 'UH', "''", 'POS', '$', '#', 'WP$', 'FW',
                              'VBD', 'VBZ'])

    # if first word POS tag is not is allowed pos tag list
    if tags[0] == 'O' and poss[0] not in ss_allowed_postags:
        valid = tokens[0] in frequent_words

    for i in range(1, len(tokens)):
        if (tags[i] == 'O' and tokens[i][0].isupper() and 'JJ' not in poss[i] and#pylint:disable=too-many-boolean-expressions
                tokens[i] not in months and tokens[i] not in days and tokens[i] not in titles):
            valid = False
    return valid

def generate_bio_entity_tags(token_list, mentions):
    """
    Generate BIO/IOB2 tags for entity detection task.
    """
    bio_tags = ['O'] * len(token_list)
    for mention in mentions:
        start = mention['start']
        end = mention['end']
        bio_tags[start] = 'B-E'
        for i in range(start + 1, end):
            bio_tags[i] = 'I-E'
    return bio_tags

def parse_single_file(input_file_path, output_file_path, frequent_word_set):
    """
    Apply HAnDS stage-III on single file.
    """
    total_sentences = 0
    valid_count = 0
    discard = 0
    with gzip.GzipFile(input_file_path, 'r') as file_i,\
            gzip.GzipFile(output_file_path, 'w') as file_o:
        for row in file_i:
            json_data = json.loads(row.decode('utf-8'))
            if is_sentence_valid(json_data, frequent_word_set):
                valid_count += 1
                json_str = json.dumps(json_data) + '\n'
                file_o.write(json_str.encode('utf-8'))
            else:
                discard += 1
            total_sentences += 1
    print('| {} | {:05.3f} | {} |'.format(discard, (discard/total_sentences)*100,
                                          total_sentences))

#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Check README.md for usage details.")
        sys.exit(1)

    word_set = read_frequent_sentence_starters(sys.argv[1], 150)
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    filepaths = list(glob.iglob(input_directory + '**/wiki_*'))
    for filepath in filepaths:
        print(filepath)
        subdir = os.path.basename(os.path.split(filepath)[0])
        basename = os.path.split(filepath)[1]
        ensure_directory(output_directory + subdir)
        output_path = output_directory + subdir + '/' + basename
        if os.path.isfile(output_path):
            continue
        parse_single_file(filepath, output_path, word_set)
