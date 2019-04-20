"""
Contains various helper function related to CADS framework.
"""
import os
import errno
import pickle
from nameparser import HumanName

#pylint:disable=raising-bad-type
def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise exception
#pylint:enable=raising-bad-type

def prepare_dictionaries(master_dictionary_path, mapping_file):
    """
    Load necessary dictionary.
    """
    md = pickle.load(open(master_dictionary_path, 'rb'))#pylint:disable=invalid-name
    #pylint:enable=invalid-name
    redirect_to_title = {}
    surface_names_to_titles = {}
    for title in md:
        for redirect in md[title]['alternate_titles']:
            redirect_to_title[redirect] = title
        if 'surface_names' in md[title]:
            for name in md[title]['surface_names']:
                if name not in surface_names_to_titles:
                    surface_names_to_titles[name] = set()
                surface_names_to_titles[name].add(title)

    mapping = {}
    with open(mapping_file) as file_p:
        for row in filter(None, file_p.read().split('\n')):
            if len(row.split('\t')) == 2:
                fb_label, figer_label = row.split('\t')
                mapping[fb_label] = figer_label
            else:
                mapping[row] = row

    return {
        'md'    : md,
        'rtt'   : redirect_to_title,
        'sntt'  : surface_names_to_titles,
        'lmap'  : mapping
    }

def get_actual_title(link, dictionaries):
    """
    Return actual title of wiki page link if exists.
    """
    if link not in dictionaries['md']:
        if link not in dictionaries['rtt']:
            link = None
        else:
            link = dictionaries['rtt'][link]
    return link

def map_labels(labels, mapping):
    """
    Map a list of labels with dictionary values if present,
    """
    mapped_labels = set()
    for label in [mapping[x] for x in labels if mapping.get(x, 0)]:
        mapped_labels.add(label)
    return mapped_labels

def lowercase_support_confidence(link, dictionaries):
    """
    Compute support and confidence of lowercase surface names.
    """
    link = get_actual_title(link, dictionaries)
    surface_names = dictionaries['md'][link]['surface_names']
    lower_case_count = 0
    total = 0
    for name in surface_names:
        if name.islower():
            lower_case_count += surface_names[name]
        total += surface_names[name]
    return lower_case_count, (lower_case_count / total) * 100

def is_title_entity(link, dictionaries):
    """
    Return status of link.
        0   : Not entity
        1   : Entity
        -1  : Couldn't infer. Discard.
    """
    link = get_actual_title(link, dictionaries)
    if not link:
        return -1

    labels = dictionaries['md'][link]['labels']
    mapped_labels = map_labels(labels, dictionaries['lmap'])
    if mapped_labels:
        return 1

    # A special case
    if '/government/political_ideology' in labels:
        return 0

    try:
        support, confidence = lowercase_support_confidence(link, dictionaries)
    except KeyError:
        return -1


    if support > 50 and confidence > 50:
        return 0

    return -1

def parse_title(title, dictionaries):
    """
    Parse a Wikipedia page title or redirect.
    """
    correct_names = set()
    label_set = set(dictionaries['md'][get_actual_title(title, dictionaries)].get('labels', ''))
    title = title.replace('_', ' ')

    # remove part of title that are in round brackets. Eg. Rice (novel)
    filtered_title = title[:title.find('(')].strip() if '(' in title else title

    # if first character is '(', then do not remove round brackets
    if not filtered_title:
        filtered_title = title

    correct_names.add(filtered_title)
    correct_names.add(filtered_title.lower())

    # If type person, add first and last name
    if '/people/person' in label_set and '/organization' not in label_set\
    and '/location' not in label_set:
        name = HumanName(filtered_title)
        correct_names.add(name.first)
        correct_names.add(name.last)
    return correct_names

def generate_candidate_names(title, dictionaries):
    """
    Return a set of possible referent names of a Wikipedia title.
    """
    names = set()
    names.update(parse_title(title, dictionaries))

    # add redirect names
    for redirect in dictionaries['md'][title]['alternate_titles']:
        names.update(parse_title(redirect, dictionaries))
    return set(filter(None, names))

def is_surface_name_referential(name, link, dictionaries):
    """
    Returns true, if the surface name matches with the candidate names.
    """
    title = get_actual_title(link, dictionaries)
    candidate_names = generate_candidate_names(title, dictionaries)

    if name in candidate_names or name.lower() in candidate_names:
        return True
    return False

def separate_links(link_set, dictionaries):
    """
    Partition link_set into entity and non entity set.
    """
    entities = set()
    non_entities = set()
    discard_entities = set()
    for link in link_set:
        status = is_title_entity(link, dictionaries)
        if status == 1:
            entities.add(get_actual_title(link, dictionaries))
        elif status == 0:
            non_entities.add(get_actual_title(link, dictionaries))
        else:
            discard_entities.add(link)
    return (entities, non_entities, discard_entities)

def is_entity_lowercase_dominant(link, dictionaries):
    """
    Return true if in more than 50% surface names,
    entity in expressed as lowercase words.
    """
    try:
        names = dictionaries['md'][get_actual_title(link, dictionaries)]['surface_names']
    except KeyError:
#        print(link)
        return False
    total = 0
    lowercase = 0
    for name in names:
        if name.islower():
            lowercase += names[name]
        total += names[name]
    if (lowercase / total) > 0.5:
        return True
    return False

def return_mapped_labels(link, dictionaries):
    """
    Return label set assigned by KB.
    """
    title = get_actual_title(link, dictionaries)
    if title is None:
        return set()
    labels = dictionaries['md'][title].get('labels', set())
    mapped_labels = map_labels(labels, dictionaries['lmap'])
    return mapped_labels

def return_parent_label(label):
    """
    Example:
    Input: /education/department
    Output: /education
    """
    if label.count('/') > 1:
        return label[0:label.find('/', 1)]
    return label
