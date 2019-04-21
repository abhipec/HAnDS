"""
Convert json data shared by Ren et al. to TFRecord format for entity mention classification task.
Usage:
    json_to_tfrecord prepare_local_variables <input_file_path> <word_vector_file> <unk_token>
                     <output_directory> [--normalize_digits] [--lowercase] [--transfer_learning]
                     [--tl_local_variables=<file_path>]
    json_to_tfrecord afet_data <output_directory> <input_file_path> [--test_data]
    json_to_tfrecord (-h | --help)

Options:
    -h, --help                              Display help message.
    <word_vector_file>                      Word vector file path in plain text format.
    <unk_token>                             Word representing unk_token.
    <output_directory>                      Directory where created dataset will be stored.
    <max_character_sequence_length>         Maximum characters allowed in a word.
    --normalize_digits                      If present, all digits will be replaced with 'X'
                                            individually.
    --lowercase                             If present, all words will be converted to lower case.
    --test_data                             If present, sentences with more than 100 words
                                            will not be removed. This is required if testing data
                                            has longer sentences.
    --transfer_learning                     If true, set the tl_local_variables parameter to specify
                                            the source of transfer learning.
    --tl_local_variables=<file_path>        The file path of the source data local variables.

"""
import sys
import os
import collections
import json
import re
import gzip
import pickle
from docopt import docopt
import numpy as np
#pylint:disable=no-member
import tensorflow as tf
from joblib import Parallel, delayed
sys.path.insert(0, 'utils')

def chunks(sentences, number_of_sentences):
    """
    Split a list into N sized chunks.
    """
    number_of_sentences = max(1, number_of_sentences)
    return [sentences[i:i+number_of_sentences]
            for i in range(0, len(sentences), number_of_sentences)]

def words_and_characters_to_use(filepath, lowercase, normalize_digits):
    """
    Read words and characters to use for token_found file.
    """
    words = {}
    char_list = []
    use_gzip = True if filepath[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(filepath, 'r')
    else:
        file_i = open(filepath, encoding='utf-8')
    for row in file_i:
        if use_gzip:
            json_data = json.loads(row.decode('utf-8'))
        else:
            json_data = json.loads(row)
        tokens = json_data['tokens']
        for token in tokens:
            if lowercase:
                new_token = token.lower()
            else:
                new_token = token
            if normalize_digits:
                #pylint:disable=anomalous-backslash-in-string
                new_token = re.sub("\d", "X", new_token)
            if not words.get(new_token):
                words[new_token] = 0
            words[new_token] += 1

        try:
            entity_mentions = json_data['mentions']
        except KeyError:
            entity_mentions = json_data['links']
        for mention in entity_mentions:
            for char in ' '.join(tokens[mention['start']:mention['end']]):
                char_list.append(char)
    file_i.close()
    return words, char_list

def load_embeddings(filepath, words_to_load):
    """
    Load selected word vectors based on word list.
    """
    word_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file_p:
        for line in file_p:
            splits = line.split(' ')
            if splits[-1] == '\n':
                del splits[-1]
            word = splits[0]
            if word in words_to_load:
                word_dict[word] = np.array([float(x) for x in splits[1:]],
                                           dtype=np.core.numerictypes.float32)

    # enumeration will start from 1
    word_to_num = dict(zip(word_dict.keys(), range(1, len(word_dict) + 1)))

    # 0th pre_trained_embedding will be remain 0
    # assumption word 'the' will be in every pretrained word vector.
    pre_trained_embeddings = np.zeros((len(word_to_num) + 1, len(word_dict['the'])),
                                      dtype=np.core.numerictypes.float32
                                     )
    for word in word_to_num:
        pre_trained_embeddings[word_to_num[word]] = word_dict[word]
    return word_to_num, pre_trained_embeddings

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def make_character_to_num(characters, remove_below):
    """
    Convert character list into a dictionary.
    """
    counter = collections.Counter(characters)

    chrs, counts = list(zip(*sorted(counter.items(), key=lambda x: (-x[1], x[0]))))

    chrs = np.array(chrs)
    counts = np.array(counts)

    # remove infrequent characters
    mask = counts >= remove_below
    chrs = chrs[mask]
    counts = counts[mask]

    # 0th character will be used as padding
    num_to_chrs = dict(enumerate(chrs, 1))

    # add unique character
    num_to_chrs[len(num_to_chrs) + 1] = 'unk'
    # add end of mention character
    num_to_chrs[len(num_to_chrs) + 1] = 'eos'

    chrs_to_num = invert_dict(num_to_chrs)

    return chrs_to_num

def generate_labels_to_numbers(filepath):
    """
    Generate label to number dictionary.
    """
    label_set = set()
    use_gzip = True if filepath[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(filepath, 'r')
    else:
        file_i = open(filepath, encoding='utf-8')
    for row in file_i:
        if use_gzip:
            json_data = json.loads(row.decode('utf-8'))
        else:
            json_data = json.loads(row)
        try:
            entity_mentions = json_data['mentions']
        except KeyError:
            entity_mentions = json_data['links']
        for mention in entity_mentions:
            label_set.update(mention['labels'])

    label_to_number = dict(zip(list(label_set), range(len(label_set))))
    file_i.close()
    return label_to_number

def labels_status(labels):
    """
    Check is labels is clean or not.
    """
    if not labels:
        return 1
    leaf = max(labels, key=lambda x: x.count('/'))
    clean = 1
    for label in labels:
        if label not in leaf:
            clean = 0
    return clean

def get_uid(json_data, start=None, end=None):
    """
    Give a unique id using document, paragraph and sentence number.
    """
    if 'did' in json_data:
        did = json_data['did']
    else:
        did = json_data['fileid']

    pid = json_data.get('pid', '')
    if 'sid' in json_data:
        sid = json_data['sid']
    else:
        sid = json_data['senid']

    if start or end:
        uid = '_'.join([str(did), str(pid), str(sid), str(start), str(end)])
    else:
        uid = '_'.join([str(did), str(pid), str(sid)])
    return uid

def process_sentences(l_vars, arguments, sentences, random_id, use_gzip):
    """
    Process a subset of sentences.
    """
    writer = tf.python_io.TFRecordWriter(
        arguments['<output_directory>']
        + os.path.basename(arguments['<input_file_path>']) + '_' + str(random_id)  + '.tfrecord',
        options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    )

    for row in sentences:
        if use_gzip:
            json_data = json.loads(row.decode('utf-8'))
        else:
            json_data = json.loads(row)
        tokens = json_data['tokens']

        discard = False
        if len(tokens) > 100 and not arguments['--test_data']:
            print('Tokens length can not be greater than 100')
            discard = True

        new_tokens = []
        for token in tokens:
            new_token = token
            if arguments['--lowercase']:
                new_token = new_token.lower()

            if arguments['--normalize_digits']:
                #pylint:disable=anomalous-backslash-in-string
                new_token = re.sub("\d", "X", new_token)
            new_tokens.append(new_token)

        try:
            entity_mentions = json_data['mentions']
        except KeyError:
            entity_mentions = json_data['links']
        for mention in entity_mentions:
            start = mention['start']
            end = mention['end']

            uid = bytes(get_uid(json_data, start, end), 'utf-8')
            # lc and rc include mention

            left_context = new_tokens[:end]
            entity = tokens[start:end]
            right_context = new_tokens[start:]

            ex = tf.train.SequenceExample()

            ex.context.feature["uid"].bytes_list.value.append(uid)
            ex.context.feature["lcl"].int64_list.value.append(len(left_context))
            ex.context.feature["rcl"].int64_list.value.append(len(right_context))
            ex.context.feature["eml"].int64_list.value.append(len(' '.join(entity)) + 1)
            ex.context.feature["clean"].int64_list.value.append(labels_status(mention['labels']))

            lc_ids = ex.feature_lists.feature_list["lci"]
            rc_ids = ex.feature_lists.feature_list["rci"]
            em_ids = ex.feature_lists.feature_list["emi"]
            label_list = ex.feature_lists.feature_list["labels"]

            for word in left_context:
                lc_ids.feature.add().int64_list.value.append(
                    l_vars['wtn'].get(word, l_vars['wtn']['unk']))

            for word in right_context:
                rc_ids.feature.add().int64_list.value.append(
                    l_vars['wtn'].get(word, l_vars['wtn']['unk']))

            for char in ' '.join(entity):
                em_ids.feature.add().int64_list.value.append(
                    l_vars['ctn'].get(char, l_vars['ctn']['unk']))

            em_ids.feature.add().int64_list.value.append(l_vars['ctn']['eos'])

            temp_labels = [0] * len(l_vars['ltn'])
            for label in mention['labels']:
                if label in l_vars['ltn']:
                    temp_labels[l_vars['ltn'][label]] = 1
            for label in temp_labels:
                label_list.feature.add().int64_list.value.append(label)

            if not discard:
                writer.write(ex.SerializeToString())
            else:
                print(tokens)
    writer.close()

def convert_afet(l_vars, arguments):
    """
    Conert data shared by AFET paper.
    """
    input_file_path = arguments['<input_file_path>']
    use_gzip = True if input_file_path[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(input_file_path, 'r')
        sentences = list(filter(None, file_i.read().split(b'\n')))
    else:
        file_i = open(input_file_path, encoding='utf-8')
        sentences = list(filter(None, file_i.read().split('\n')))
    parts = chunks(sentences, 10000)
    Parallel(n_jobs=8)(delayed(process_sentences)
                       (l_vars, arguments, parts[uid], uid, use_gzip) for uid in range(len(parts)))
    file_i.close()

def prepare_or_load_variables(arguments):
    """
    If existing variables exists, load it.
    Otherwise prepare.
    """
    output_directory = arguments['<output_directory>']
    if os.path.isfile(output_directory + 'local_variables.pickle'):
        print('Using pre saved local variables.')
        l_vars, arguments = pickle.load(open(output_directory + 'local_variables.pickle', 'rb'))
    elif arguments['--transfer_learning']:
        # All local variables will be same as that of the specified file path except the
        # label to number dictionary.
        l_vars, _ = pickle.load(open(arguments['--tl_local_variables'], 'rb'))
        label_to_num = generate_labels_to_numbers(arguments['<input_file_path>'])
        l_vars['ltn'] = label_to_num
        # writing local variables and the arguments used
        arguments.pop('prepare_local_variables', None)
        pickle.dump((l_vars, arguments),
                    open(arguments['<output_directory>'] + 'local_variables.pickle', 'wb'))
    else:
        print('Generating unique words and characters.')
        words_dict, characters_to_load = words_and_characters_to_use(
            arguments['<input_file_path>'],
            arguments['--lowercase'],
            arguments['--normalize_digits'])

        words_dict[arguments['<unk_token>']] = 1

        print('Total words to load: ', len(words_dict))

        print('Loading word embeddings.')
        word_to_num, embedding = load_embeddings(arguments['<word_vector_file>'], words_dict)
        print('Embedding shape', embedding.shape)

        print('Percentage of unique words found: ', (len(word_to_num)/len(words_dict)) * 100)
        total_words = sum(words_dict.values())
        total_words_found = 0
        for word in word_to_num:
            total_words_found += words_dict[word]

        print('Percentage of total words found: ', (total_words_found / total_words) * 100)
        char_to_num = make_character_to_num(characters_to_load, 5)
        label_to_num = generate_labels_to_numbers(arguments['<input_file_path>'])
        l_vars = {
            'wtn' : word_to_num,
            'ltn' : label_to_num,
            'ctn' : char_to_num,
            'word_embedding' : embedding
        }
        # writing local variables and the arguments used
        arguments.pop('prepare_local_variables', None)
        pickle.dump((l_vars, arguments),
                    open(arguments['<output_directory>'] + 'local_variables.pickle', 'wb'))

    return l_vars, arguments

#pylint:disable=invalid-name
if __name__ == '__main__':
    cmd_arguments = docopt(__doc__)
    # load the local variables and parameters used to create such variables
    local_variables, parameters = prepare_or_load_variables(cmd_arguments)

    if cmd_arguments['afet_data']:
        parameters['<input_file_path>'] = cmd_arguments['<input_file_path>']
        parameters['--test_data'] = cmd_arguments['--test_data']
        print(parameters)
        convert_afet(local_variables, parameters)
