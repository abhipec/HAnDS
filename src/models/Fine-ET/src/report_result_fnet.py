"""
Report results of entity classification in terms of evaluation metrics defined in [1].

[1] Ling, Xiao, and Daniel S. Weld. “Fine-Grained Entity Recognition.” In AAAI, 2012.
http://xiaoling.github.io/pubs/ling-aaai12.pdf.
"""

import sys
import json

def return_parent_label(label_name):
    """
    Example:
    Input: /education/department
    Output: /education
    """
    if label_name.count('/') > 1:
        return label_name[0:label_name.find('/', 1)]
    return label_name

def f1_score(precision, recall):
    """
    Compute f1 score.
    """
    if precision or recall:
        return 2 * precision * recall / (precision + recall)
    return 0

def strict_score(gold, prediction):
    """
    Compute the strict evaluation metric.
    """
    intersection_uids = set(gold.keys()).intersection(set(prediction.keys()))
    equal_count = 0
    for key in intersection_uids:
        if gold[key] == prediction[key]:
            equal_count += 1
    precision = 100 * equal_count / len(prediction)
    recall = 100 * equal_count / len(gold)
    print('Strict score')
    print('{:05.2f} {:05.2f} {:05.2f}'.format(precision, recall, f1_score(precision, recall)))

def loose_macro(gold, prediction):
    """
    Compute the loose macro score,
    """
    count = 0
    for key in prediction:
        count += len(prediction[key].intersection(gold.get(key, set()))) / len(prediction[key])
    precision = 100 * count / len(prediction)

    count = 0
    for key in gold:
        count += len(gold[key].intersection(prediction.get(key, set()))) / len(gold[key])
    recall = 100 * count / len(gold)
    print('loose macro')
    print('{:05.2f} {:05.2f} {:05.2f}'.format(precision, recall, f1_score(precision, recall)))

def loose_micro(gold, prediction):
    """
    Compute the loose micro score.
    """

    count_n = 0
    count_d = 0
    for key in prediction:
        count_n += len(prediction[key].intersection(gold.get(key, set())))
        count_d += len(prediction[key])

    precision = 100 * count_n / count_d

    count_n = 0
    count_d = 0
    for key in gold:
        count_n += len(gold[key].intersection(prediction.get(key, set())))
        count_d += len(gold[key])

    recall = 100 * count_n / count_d

    print('loose micro')
    print('{:05.2f} {:05.2f} {:05.2f}'.format(precision, recall, f1_score(precision, recall)))

def return_only_top_level_labels(label_list):
    """
    Return only top level labels.
    """
    to_return = []
    for label_name in label_list:
        if label_name.count('/') == 1:
            to_return.append(label_name)
    return to_return

def load_label_patch(filepath):
    """
    Patch labels.
    Primary reason: Two training dataset have different labels,
    however most of the labels have one-to-one maping.
    """
    mapping = {}
    with open(filepath) as file_i:
        for line in filter(None, file_i.read().split('\n')):
            if '#' in line:
                continue
            part1, part2 = line.split(' ')
            if ',' in part1:
                part1 = frozenset(part1.split(','))
            mapping[part1] = part2.split(',')
    return mapping

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

#pylint:disable=invalid-name,no-member
if __name__ == '__main__':
    gold_result_filepath = sys.argv[1]
    predicted_result_filepath = sys.argv[2]
    top_level_only = int(sys.argv[3])
    if len(sys.argv) == 5:
        label_patch_file = sys.argv[4]
        label_patch = load_label_patch(label_patch_file)
    else:
        label_patch = {}
    gold_results = {}

    to_skip = set()
    # Entity mentions with these labels will be ignored during comparison.
#    to_skip_labels = set(['/time', '/finance/currency'])
#    to_skip_labels = set(['/sport', '/toy', '/particle', '/fictional_character',
#                          '/organization/music', '/school_subject',
#                          '/location/extraterrestrial_location'])
    to_skip_labels = set()
    with open(gold_result_filepath) as file_p:
        for row in file_p:
            json_data = json.loads(row)
            fileid = json_data.get('fileid', '')
            pid = json_data.get('pid', '')
            sid = json_data['senid']
            for mention in json_data['mentions']:
                start = mention['start']
                end = mention['end']
                uid = get_uid(json_data, start, end)
                labels = []
                if frozenset(mention['labels']) in label_patch:
                    labels = label_patch[frozenset(mention['labels'])]
                else:
                    for label in mention['labels']:
                        if label in label_patch:
                            labels += label_patch[label]
                        else:
                            labels.append(label)
                new_labels = set()
                for label in labels:
                    new_labels.add(label)
                    new_labels.add(return_parent_label(label))
                labels = new_labels
                if labels.intersection(to_skip_labels):
                    to_skip.add(uid)
                    continue
                if top_level_only:
                    gold_results[uid] = set(return_only_top_level_labels(labels))
                else:
                    gold_results[uid] = labels

    predicted_results = {}
    with open(predicted_result_filepath) as file_p:
        rows = filter(None, file_p.read().split('\n'))
        for row in rows:
            uid, labels = row.split('\t')
            if uid in to_skip:
                continue
            labels = labels.split(',')
            if top_level_only:
                predicted_results[uid] = set(return_only_top_level_labels(labels))
            else:
                predicted_results[uid] = set(labels)
    
    strict_score(gold_results, predicted_results)
    loose_macro(gold_results, predicted_results)
    loose_micro(gold_results, predicted_results)
