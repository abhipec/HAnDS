"""
Code to parse TypeNet.
"""
import sys
import networkx as nx

def parse_typenet_structure(file_path):
    """
    Parse the TypeNet.
    """
    mapping_local = {}
    di_graph_local = nx.DiGraph()
    with open(file_path) as file_i:
        for row in file_i:
            if '->' in row:
                left_side, right_side = row.split('->')
                left_side = left_side.strip()
                right_side = right_side.strip()
                di_graph_local.add_edge(right_side, left_side)
                if 'Synset' not in left_side:
                    if left_side not in mapping_local:
                        mapping_local[left_side] = set()
                    mapping_local[left_side].add(right_side)
    return mapping_local, di_graph_local

def load_figer_set_present_in_conll(file_path):
    """
    Load the freebase types that were present in conll dataset.
    """
    types_covered_local = set()
    with open(file_path) as file_i:
        for row in file_i:
            fb_type, _ = row.split('\t')
            types_covered_local.add(fb_type)
    return types_covered_local

def parse_typenet_mapping(file_path):
    """
    Parse the TypeNet.
    """
    mapping_local = {}
    with open(file_path) as file_i:
        for row in filter(None, file_i.read().split('\n\n')):
            annotations = row.split('\n')
            if annotations[0] not in mapping_local:
                mapping_local[annotations[0]] = set()
            mapping_local[annotations[0]].update(annotations[1:])
    return mapping_local

#pylint:disable=invalid-name
if __name__ == '__main__':
    mapping_structural, graph = parse_typenet_structure(sys.argv[1])
    # 1081 Types in TypeNet
#    mapping = parse_typenet_mapping(sys.argv[1])
    types_covered = load_figer_set_present_in_conll(sys.argv[2])
    # Label coverage analysis of FIGER and CoNLL is manual.
    # Person: 18 (including animal and god)
    # Location: 24 (as per figer_type_in_conll_location.txt)
    # Organization: 22 (as per figer_type_in_conll_organization.txt)
    # Out of scope: 15
    # Misc: 98 - 18 - 24 - 22 = 34
    types_covered.add("Synset('product.n.02')")
    types_covered.add("Synset('vehicle.n.01')")
    types_covered.add("Synset('event.n.01')")
    types_covered.add("Synset('equipment.n.01')")
    types_covered.add("Synset('imaginary_being.n.01')")
    types_covered.add("Synset('video.n.01')")
    # Type coverage of TypeNet is calculated automatically by using the FIGER freebase mapping.
    # Loop over all TypeNet types
    with open(sys.argv[3], 'w') as file_p:
        present, not_present = 0, 0
        for label in mapping_structural:
            ancesostors = nx.ancestors(graph, label)
            ancesostors.add(label)
            if not types_covered.intersection(ancesostors):
                not_present += 1
                file_p.write(label + '\n')
            else:
                present += 1
    # Label coverage analysis of TypeNet and CoNLL is manual.
    print('Out of scope:', not_present) #268
    person_nodes = ["Synset('spiritual_being.n.01')", "Synset('imaginary_being.n.01')",
                    "/people/person"]
    person_set = set()
    for label in mapping_structural:
        for person_node in person_nodes:
            if label in nx.descendants(graph, person_node):
                person_set.add(label)
    print('Person:', len(person_set)) #257
    location_nodes = ["/location/location", "Synset('facility.n.01')"]
    location_set = set()
    for label in mapping_structural:
        for location_node in location_nodes:
            if label in nx.descendants(graph, location_node):
                location_set.add(label)
    print('Location:', len(location_set)) #197
    organization_nodes = ["Synset('group.n.01')", "Synset('facility.n.01')"]
    organization_set = set()
    for label in mapping_structural:
        for organization_node in organization_nodes:
            if label in nx.descendants(graph, organization_node):
                organization_set.add(label)
    print('Organization:', len(organization_set)) #188
    print("Misc:",
          1081 - len(person_set) - len(organization_set) - len(location_set) - not_present) #171
