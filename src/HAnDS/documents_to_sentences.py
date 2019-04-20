"""
Convert document oriented data to sentences.
"""
import sys
import os
import glob
import gzip
import json
from helper import ensure_directory

#pylint:disable=too-many-locals
def map_links(para_links, sentence, para_text):
    """
    Map character offsets to word offsets.
    """
    # All links in para offset that are within these limits must be
    # present in this sentence.
    sentence_begin_char_offset = sentence['characterOffsetBegin'][0]
    sentence_end_char_offset = sentence['characterOffsetEnd'][-1]

    para_links_to_search = []
    for para_link in para_links:
        if (para_link['start'] >= sentence_begin_char_offset
                and para_link['end'] <= sentence_end_char_offset):
            para_links_to_search.append(para_link)

    mapping = []
    valid = 1
    for para_link in para_links_to_search:
        char_offset_start_list = sentence['characterOffsetBegin']
        char_offset_end_list = sentence['characterOffsetEnd']
        try:
            start = char_offset_start_list.index(para_link['start'])
            end = char_offset_end_list.index(para_link['end'])
            mapping.append({
                'start' : start,
                'end' : end + 1,
                'link' : para_link['link'],
                'name' : para_link['name']
            })
        except ValueError:
            end_offset = para_link['end']
            start_offset = para_link['start']
            suffix = para_text[end_offset:para_text.find(' ', end_offset - 1)]
            prefix = para_text[para_text[:start_offset].rfind(' ') + 1:start_offset]
            corrected = False
            try:
                if not prefix and suffix:
                    if suffix == 's' or suffix == 's.' or suffix == 's,':
                        end = char_offset_end_list.index(para_link['end'] + 1)
                        corrected = True
                    if suffix[0] == '-':
                        end = char_offset_end_list.index(para_link['end'] + len(suffix))
                        corrected = True
                    start = char_offset_start_list.index(para_link['start'])
                if not suffix and prefix:
                    if prefix[-1] == '-':
                        start = char_offset_start_list.index(para_link['start'] - len(prefix))
                        end = char_offset_end_list.index(para_link['end'])
                        corrected = True
            except ValueError:
                print(para_link)
                print(suffix)
                print(prefix)
                print(' '.join(sentence['tokens']))
                corrected = False
            if corrected:
                mapping.append({
                    'start' : start,
                    'end' : end + 1,
                    'link' : para_link['link'],
                    'name' : para_link['name']
                })
                valid = 1
            else:
                valid = 0
    return mapping, valid

def parse_single_file(input_file_path, output_file_path):#pylint:disable=too-many-locals
    """
    Convert document level annotation of a single file to sentence level annotations.
    """
    #pylint:enable=too-many-locals
    with gzip.GzipFile(input_file_path, 'r') as file_i,\
            gzip.GzipFile(output_file_path, 'w') as file_o:
        valid_count = 0
        total_count = 0
        for row in file_i:
            json_data = json.loads(row.decode('utf-8'))
            document_id = json_data['did']
            for pid, paragraph in enumerate(json_data['paragraphs']):
                text = paragraph[0]
                links = paragraph[1]
                sentences = paragraph[2]
                for sentence in sentences:
                    mapped_links, valid = map_links(links, sentence, text)
                    valid_count += valid
                    total_count += 1
                    tagged_sentence = {
                        'did' : document_id,
                        'pid' : pid,
                        'pos': sentence['pos'],
                        'sid': sentence['sid'],
                        'tokens': sentence['tokens'],
                        'links': mapped_links
                    }
                    json_str = json.dumps(tagged_sentence) + '\n'
                    file_o.write(json_str.encode('utf-8'))

#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python utils/wikification/documents_to_sentences.py',
              '../data/processed/wikification/input_directory/',
              '../data/processed/wikification/output_directory/')
        sys.exit(1)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    filepaths = list(glob.iglob(input_directory + '**/wiki_*'))
    for filepath in filepaths:
        print(filepath)
        subdir = os.path.basename(os.path.split(filepath)[0])
        basename = os.path.split(filepath)[1]
        ensure_directory(output_directory + subdir)
        output_path = output_directory + subdir + '/' + basename
        if os.path.isfile(output_path):
            continue
        parse_single_file(filepath, output_path)
