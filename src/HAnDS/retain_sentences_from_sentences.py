"""
Retain sentences from the corpus generated using already sampled sentences.
"""


import sys
import glob
import json
import gzip

def retain_sentences(input_directory,#pylint:disable=too-many-locals
                     sentences_to_retain_file,
                     output_file_path):
    """
    Retain desired sentences from the generated datasets.
    """
    use_gzip = True if sentences_to_retain_file[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(sentences_to_retain_file, 'r')
    else:
        file_i = open(sentences_to_retain_file, encoding='utf-8')

    sentences = set()
    for row in file_i:
        if use_gzip:
            json_data = json.loads(row.decode('utf-8'))
        else:
            json_data = json.loads(row)
        uid = '_'.join([
            str(json_data['did']),
            str(json_data['pid']),
            str(json_data['sid'])])
        sentences.add(uid)


    with gzip.GzipFile(output_file_path, 'w') as file_o:
        for file in glob.iglob(input_directory + '**/wiki_*'):
            print(file)
            with gzip.GzipFile(file, 'r') as file_p:
                for row in file_p:
                    json_data = json.loads(row.decode('utf-8'))
                    uid = '_'.join([
                        str(json_data['did']),
                        str(json_data['pid']),
                        str(json_data['sid'])])
                    if uid in sentences:
                        json_str = json.dumps(json_data) + '\n'
                        file_o.write(json_str.encode('utf-8'))


if __name__ == '__main__':
    retain_sentences(sys.argv[1], sys.argv[2], sys.argv[3])
