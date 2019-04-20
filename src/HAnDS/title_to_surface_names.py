"""
Extracts info link urls and surface forms from Wikiextractor output.
"""

import sys
import glob
import pickle
from bs4 import BeautifulSoup
from url_conversion_helper import sanitize_url

__author__ = "Abhishek, Sanya B. Taneja, and Garima Malik"
__maintainer__ = "Abhishek"

def reverse_dictionary(title_to_freebase):
    """
    Create a redirect to title dictionary.
    """
    redirect_to_title = {}
    for link in title_to_freebase:
        for redirect in title_to_freebase[link]['alternate_titles']:
            redirect_to_title[redirect] = link
    return redirect_to_title

def process_file(input_file_path, title_to_freebase, redirect_to_title):
    """
    Parse a single html file.
    """

    file_i = open(input_file_path, 'r', encoding='utf-8')
    soup = BeautifulSoup(file_i, "html.parser")
    not_found = 0
    found_redirect = 0
    found = 0
    for link in soup.find_all('a'):
        text = str(link.string)
        title = str(link.get('href'))
        link_title = sanitize_url(title)

        if not link_title:
            continue

        link_title = link_title.replace(' ', '_')

        if link_title in title_to_freebase:
            if 'surface_names' not in title_to_freebase[link_title]:
                title_to_freebase[link_title]['surface_names'] = {}
            if text not in title_to_freebase[link_title]['surface_names']:
                title_to_freebase[link_title]['surface_names'][text] = 0
            title_to_freebase[link_title]['surface_names'][text] += 1
            found += 1
        else:
            if link_title in redirect_to_title:
                key = redirect_to_title[link_title]
                if 'surface_names' not in title_to_freebase[key]:
                    title_to_freebase[key]['surface_names'] = {}
                if text not in title_to_freebase[key]['surface_names']:
                    title_to_freebase[key]['surface_names'][text] = 0
                title_to_freebase[key]['surface_names'][text] += 1
                found_redirect += 1
            else:
                not_found += 1
    print(not_found, found_redirect, found)

#pylint:disable=invalid-name
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Arguments missing.')
        print('arg1: Output directory of Wikiextractor.')
        print('arg2: title_to_fbmid pickle.')
        print('arg3: Output file_path.')
        sys.exit(1)
    title_to_freebase_dict = pickle.load(open(sys.argv[2], 'rb'))
    redirect_to_title_dict = reverse_dictionary(title_to_freebase_dict)
    files = list(glob.iglob(sys.argv[1] + '**/wiki_*'))
    print('Starting')
    for file in files:
        process_file(file, title_to_freebase_dict, redirect_to_title_dict)
        print(file)
    #pickle dump
    pickle.dump(title_to_freebase_dict, open(sys.argv[3], 'wb'))
