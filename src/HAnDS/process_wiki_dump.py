"""
Pre-process a Wiki extractor files and segment sentences with CoreNLP.
"""
import os
import sys
import glob
import gzip
import json
import warnings
from multiprocessing import Process
from bs4 import BeautifulSoup
from url_conversion_helper import sanitize_url
from corenlp import CoreNlPClient
from helper import ensure_directory

__author__ = "Abhishek, Sanya B. Taneja, and Garima Malik"
__maintainer__ = "Abhishek"

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def chunks(sentences, number_of_sentences):
    """
    Split a list into N sized chunks.
    """
    number_of_sentences = max(1, number_of_sentences)
    return [sentences[i:i+number_of_sentences]
            for i in range(0, len(sentences), number_of_sentences)]

def flatten_json_with_key(data, json_key):
    """
    Return a list of tokens from a list of json obj.
    """
    to_return = []
    for obj in data:
        to_return.append(obj[json_key])
    return to_return

def paragraph_to_sentences(paragraph, client):
    """
    Convert paragraph with char offsets to sentences with word offsets.
    """
    output = client.annotate(paragraph.encode('utf-8'))
    if not output:
        return None
    tagged_sentences = []
    for sentence in output['sentences']:
        pos_tags = flatten_json_with_key(sentence['tokens'], 'pos')
        tokens = flatten_json_with_key(sentence['tokens'], 'word')
        #pylint:disable=invalid-name
        characterOffsetBegin = flatten_json_with_key(sentence['tokens'], 'characterOffsetBegin')
        characterOffsetEnd = flatten_json_with_key(sentence['tokens'], 'characterOffsetEnd')
        tagged_sentences.append({
            'pos': pos_tags,
            'sid': sentence['index'],
            'tokens': tokens,
            'characterOffsetBegin': characterOffsetBegin,
            'characterOffsetEnd': characterOffsetEnd
        })
    return tagged_sentences

#pylint:disable=too-many-branches
def paragraphs_and_outlinks(wiki_file_path):
    """
    Process single wiki file, yield single documents.
    """
    with open(wiki_file_path, 'r', encoding='utf-8') as file_i:
        paragraphs_with_offsets = []
        outgoing_links = set()
        document_id = ''
        for line in file_i:
            #end of an old doc
            if line.startswith("</doc>"):
                yield paragraphs_with_offsets, outgoing_links, document_id

            #beginning of a new doc
            if line.startswith("<doc"):
                soup = BeautifulSoup(line, "html.parser")
                document_id = soup.contents[0].get('id', 0)
                document_title = soup.contents[0].get('title', '')
                if not document_id:
                    print(document_title)
                document_title = document_title.replace(' ', '_')
                paragraphs_with_offsets = []
                outgoing_links = set([document_title])
                continue

            if line == '\n':
                continue

            soup = BeautifulSoup(line, "html.parser")
            para_string = ''
            offset = 0
            output_offset = []
            for content in soup.contents:
                string = content.string
                if not string:
                    continue

                # if first character of a para string is a space
                if not para_string and string[0] == ' ':
                    string = string.lstrip()

                if not string:
                    continue

                if string[0] == '.':
                    string = ' .' + string[1:]


                #if string was contained inside a <a> tag, consider it to be an entity
                if content.name == "a":
                    try:
                        link = sanitize_url(content["href"])
                    except KeyError:
                        link = None
                    string = string.strip()
                    if link:
                        link = link.replace(' ', '_')
                        output_offset.append({
                            'start': offset,
                            'end': offset + len(string),
                            'link': link,
                            'name': string
                        })
                        outgoing_links.add(link)

                para_string += string
                offset += len(string)
            # ignore small paragraphs
            if offset < 100:
                continue
            # remove trailing \n
            if para_string[-1] == '\n':
                para_string = para_string[:-1]
            paragraphs_with_offsets.append((para_string, output_offset))

def process_paragraph(para_string, client):
    """
    Annotate a single paragraph.
    """
    return paragraph_to_sentences(para_string, client)

def process_article(paragraph_list, client):
    """
    Annotate a single article.
    """
    for _, (paragraph, offsets) in enumerate(paragraph_list):
        sentences = process_paragraph(paragraph,
                                      client)
        if not sentences:
            continue
        yield sentences, paragraph, offsets

def parse_single_file(input_file_path, output_file_path):
    """
    Convert a single file to sentences.
    """
    client = CoreNlPClient()
    with gzip.GzipFile(output_file_path, 'w') as file_o:
        for paragraphs, links, doc_id in paragraphs_and_outlinks(input_file_path):
            article = {}
            article['did'] = doc_id
            article['outgoing_links'] = list(links)
            processed_paragraphs = []
            for sentences, paragraph, offsets in process_article(paragraphs, client):
                processed_paragraphs.append((paragraph, offsets, sentences))
            article['paragraphs'] = processed_paragraphs
            json_str = json.dumps(article) + '\n'
            json_bytes = json_str.encode('utf-8')
            file_o.write(json_bytes)

def parse_multipe_files(input_filepaths, output_file_directory):
    """
    Parse multiple input files serially.
    """
    for input_filepath in input_filepaths:
        print(input_filepath)
        subdir = os.path.basename(os.path.split(input_filepath)[0])
        basename = os.path.split(input_filepath)[1]
        ensure_directory(output_file_directory + subdir)
        output_path = output_file_directory + subdir + '/' + basename + '.json.gz'
        if os.path.isfile(output_path):
            continue
        parse_single_file(input_filepath, output_path)

#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python utils/wikification/process_wiki_dump.py',
              '../data/processed/extracted_dump/',
              '../data/processed/wikification/cads_2_processed_wiki/')
        sys.exit(1)
    wikiextractor_directory = sys.argv[1]
    output_directory = sys.argv[2]

    filepaths = list(glob.iglob(wikiextractor_directory + '**/wiki_*'))
    parts = chunks(filepaths, 500)
    print(len(parts))
    processes = []
    for i, part in enumerate(parts):
        processes.append(Process(target=parse_multipe_files, args=(part, output_directory)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
