"""
Convert th CoNLL format data to TFRecord format for entity mention detection task.
Usage:
    json_to_tfrecord prepare_local_variables <input_file_path> <word_vector_file> <unk_token>
                     <output_directory> <max_character_sequence_length> [--normalize_digits]
                     [--lowercase]
    json_to_tfrecord conll_data <output_directory> <input_file_path> [--test_data]
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
"""
import sys
import os
import pickle
import re
from docopt import docopt
import numpy as np
#pylint:disable=no-member
import tensorflow as tf
from joblib import Parallel, delayed

def chunks(sentences, number_of_sentences):
    """
    Split a list into N sized chunks.
    """
    number_of_sentences = max(1, number_of_sentences)
    return [sentences[i:i+number_of_sentences]
            for i in range(0, len(sentences), number_of_sentences)]

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


def words_and_characters_to_use(filepath, lowercase, normalize_digits):
    """
    Read words and characters to use for token_found file.
    """
    words = {}
    characters = {}
    with open(filepath, encoding='utf-8') as file_p:
        for row in filter(None, file_p.read().split('\n\n')):
            tokens = [x.split(' ')[0] for x in row.split('\n')]
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
                for character in token:
                    if character not in characters:
                        characters[character] = 0
                    characters[character] += 1
    return words, characters

def make_character_to_num(characters):
    """
    Convert character set into a dictionary.
    """
    chars = list(characters)
    return dict(zip(chars, range(1, len(chars) + 1)))

#pylint:disable=too-many-locals, too-many-branches, too-many-statements
def process_sentences(l_vars, arguments, sentences, random_id):
    """
    Process a subset of sentences.
    """
    writer = tf.python_io.TFRecordWriter(
        arguments['<output_directory>']
        + os.path.basename(arguments['<input_file_path>']) + '_' + str(random_id)  + '.tfrecord',
        options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    )

    for row in sentences:
        if '\t' in row:
            tokens = [x.split('\t')[0] for x in row.split('\n')]
            tags = [x.split('\t')[-1] for x in row.split('\n')]
        else:
            tokens = [x.split(' ')[0] for x in row.split('\n')]
            tags = [x.split(' ')[-1] for x in row.split('\n')]

        discard = False
        ex = tf.train.SequenceExample()

        s_ids = ex.feature_lists.feature_list["si"]
        ex.context.feature["sl"].int64_list.value.append(len(tokens))
        c_ids = ex.feature_lists.feature_list["ci"]

        label_list = ex.feature_lists.feature_list["labels"]

        try:
            for tag in tags:
                label_list.feature.add().int64_list.value.append(l_vars['ltn'][tag])
        except KeyError:
            print(tokens)
            print(tags)
            print('Exiting, key error. Label not found.')
            sys.exit(1)

        if len(tokens) > 100 and not arguments['--test_data']:
            print('Tokens length can not be greater than 100')
            discard = True

        for token in tokens:
            new_token = token

            if arguments['--lowercase']:
                new_token = new_token.lower()

            if arguments['--normalize_digits']:
                #pylint:disable=anomalous-backslash-in-string
                new_token = re.sub("\d", "X", new_token)

            s_ids.feature.add().int64_list.value.append(l_vars['wtn'].get(
                new_token,
                l_vars['wtn'][arguments['<unk_token>']]))

            characters = list(token)
            if len(characters) > int(arguments['<max_character_sequence_length>']):
                print('Character length should not be more than 30')
                discard = True
            try:
                for char in characters:
                    # Replace character with its numeric id
                    # If not present, replace with 'unk' character.
                    if char in l_vars['ctn']:
                        c_ids.feature.add().int64_list.value.append(l_vars['ctn'][char])
                    else:
                        c_ids.feature.add().int64_list.value.append(l_vars['ctn']['unk'])
            except KeyError:
                print('Characters: ', characters, "Not found")
                discard = True
            # padding
            for _ in range(len(characters), int(arguments['<max_character_sequence_length>'])):
                c_ids.feature.add().int64_list.value.append(0)
        if not discard:
            writer.write(ex.SerializeToString())
        else:
            print(tokens)
    writer.close()

def convert_conll(l_vars, arguments):
    """
    Convert CoNLL format data using multiple processes.
    """
    with open(arguments['<input_file_path>'], encoding='utf-8') as file_p:
        sentences = list(filter(None, file_p.read().split('\n\n')))
        parts = chunks(sentences, 40000)
        Parallel(n_jobs=8)(delayed(process_sentences)
                           (l_vars, arguments, parts[uid], uid) for uid in range(len(parts)))

def prepare_or_load_variables(arguments):
    """
    If existing variables exists, load it.
    Otherwise prepare.
    """
    output_directory = arguments['<output_directory>']
    if os.path.isfile(output_directory + 'local_variables.pickle'):
        print('Using pre saved local variables.')
        l_vars, arguments = pickle.load(open(output_directory + 'local_variables.pickle', 'rb'))
    else:
        # load dbpedia label mappings
        print('Generating unique words and characters.')
        words_dict, characters_dict = words_and_characters_to_use(
            arguments['<input_file_path>'],
            arguments['--lowercase'],
            arguments['--normalize_digits'])

        words_dict[arguments['<unk_token>']] = 1

        # add a unique token in character dict
        # remove less frequent characters
        for character in list(characters_dict.keys()):
            if characters_dict[character] < 5:
                del characters_dict[character]
                print(character)
        characters_dict['unk'] = 1

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
        char_to_num = make_character_to_num(characters_dict.keys())
        label_to_num = {
            'B-E' : 0,
            'I-E' : 1,
            'O' : 2
        }
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

    if cmd_arguments['conll_data']:
        parameters['<input_file_path>'] = cmd_arguments['<input_file_path>']
        parameters['--test_data'] = cmd_arguments['--test_data']
        print(parameters)
        convert_conll(local_variables, parameters)
