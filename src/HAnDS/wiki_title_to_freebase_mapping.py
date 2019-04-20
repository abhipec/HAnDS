"""
Map Wikipedia title to Freebase mid. Also obtain its labels and redirect links.
"""
import sys
import pickle
from multiprocessing import Process, Manager
import SPARQLWrapper
from url_conversion_helper import freebase_encode_article

__author__ = "Abhishek, Sanya B. Taneja, and Garima Malik"
__maintainer__ = "Abhishek"

def chunks(sentences, number_of_sentences):
    """
    Split a list into N sized chunks.
    """
    number_of_sentences = max(1, number_of_sentences)
    return [sentences[i:i+number_of_sentences]
            for i in range(0, len(sentences), number_of_sentences)]

def get_freebase_mid(article_title, retry_count=0):
    """
    Return mid associated to a Wikipedia article link.
    """

    # Wikipedia article link

    query = ('''select distinct ?entity {?entity
             <http://rdf.freebase.com/key/wikipedia.en_title>
             "'''+ article_title + '''"} LIMIT 10''')

    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLWrapper.JSON)

    try:
        results = sparql.query().convert()
    except:#pylint:disable=bare-except
        if retry_count >= 3:
            print("FAIL REQUEST:FBMID", article_title)
            return ''
        else:
            retry_count += 1
        return get_freebase_mid(article_title, retry_count=retry_count)

    if len(results["results"]["bindings"]) >= 1: #should be a unique mid per page?
        result = results["results"]["bindings"][0]
        # mid found
        mid = result["entity"]["value"]
    else:
        # mid not found
        mid = ''
    return mid

def get_freebase_redirects(mid, retry_count=0):
    """
    Return list of redirect titles to wiki page mid.
    """
    redirect_list = []

    query = ('''prefix : <http://rdf.freebase.com/ns/>
                select distinct ?entity_label 
            { <'''+ mid +'''> <http://rdf.freebase.com/key/wikipedia.en> ?entity_label
            } LIMIT 1000''')

    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLWrapper.JSON)

    try:
        results = sparql.query().convert()
    except:#pylint:disable=bare-except
        if retry_count >= 3:
            print("FAIL REQUEST:REDIRECTS", mid)
            return []
        else:
            retry_count += 1
        return get_freebase_redirects(mid, retry_count=retry_count)

    for result in results["results"]["bindings"]:
        alternate_title = result["entity_label"]["value"]
        if '$' in alternate_title:
            alternate_title = alternate_title.replace(
                '$', '\\u').encode('utf-8').decode('unicode-escape')
        redirect_list.append(alternate_title)
    return redirect_list

def get_freebase_labels(mid, retry_count=0):
    """
    Return labels assigned by freebase to a particular mid.
    """
    labels = []

    query = ('''prefix : <http://rdf.freebase.com/ns/>
                select distinct ?entity_label 
            { <'''+ mid +'''> a ?entity_label
            } LIMIT 200''')

    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLWrapper.JSON)

    try:
        results = sparql.query().convert()
    except:#pylint:disable=bare-except
        if retry_count >= 3:
            print("FAIL REQUEST:LABELS", mid)
            return []
        else:
            retry_count += 1
        return get_freebase_labels(mid, retry_count=retry_count)

    for result in results["results"]["bindings"]:
        # print("label: ",result["entity_label"]["value"])
        labels.append(
            result["entity_label"]["value"][len('http://rdf.freebase.com/ns'):].replace('.', '/')
        )
    return labels

def process_list_of_title(article_titles, master_dict, uid):
    """
    Obtain the necessary values from Freebase for a list of titles.
    """
    count = 0
    for article_title in article_titles:
        freebase_encoded_title = freebase_encode_article(article_title)
        fbmid = get_freebase_mid(freebase_encoded_title)
        if fbmid:
            fb_labels = get_freebase_labels(fbmid)
            redirects = get_freebase_redirects(fbmid)
            master_dict[article_title] = ({
                'fbmid' : fbmid,
                'labels' : fb_labels,
                'alternate_titles' : redirects
            })
        count += 1
        if count % 5000 == 0:
            print(uid, count)

#pylint:disable=invalid-name
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python util/wikification/wiki_title_to_freebase_mapping.py',
              '../data/raw/enwiki-latest-all-titles-in-ns0',
              '../data/processed/wikification/title_to_fbmid.pickle')
        sys.exit(1)
    sparql = SPARQLWrapper.SPARQLWrapper("http://localhost:8890/sparql/")
    all_wiki_titles = []
    with open(sys.argv[1], 'r', encoding='utf-8') as file_p:
        for title in filter(None, file_p.read().split('\n')):
            all_wiki_titles.append(title)
    manager = Manager()
    title_to_freebase = manager.dict()

    parts = chunks(all_wiki_titles, 1300000)
    processes = []
    for i, part in enumerate(parts):
        processes.append(Process(target=process_list_of_title, args=(part, title_to_freebase, i)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    pickle.dump(dict(title_to_freebase), open(sys.argv[2], 'wb'), protocol=4)
