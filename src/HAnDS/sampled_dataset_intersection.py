"""
Compute intersection of two sampled datasets sentences.
"""


import sys
import glob
import json
import gzip

def get_uids(input_file_path):
    """
    Convert datasets to set of uids.
    """
    use_gzip = True if input_file_path[-3:] == '.gz' else False
    if use_gzip:
        file_i = gzip.GzipFile(input_file_path, 'r')
    else:
        file_i = open(input_file_path, encoding='utf-8')

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

    return sentences

if __name__ == '__main__':
    sentences_1 = get_uids(sys.argv[1])
    sentences_2 = get_uids(sys.argv[2])
    print(len(sentences_1))
    print(len(sentences_2))
    print(len(sentences_1.intersection(sentences_2)))
