"""
Convert CoNLL formatted text to plan text.
"""
import sys

if __name__ == '__main__':
    with open(sys.argv[1]) as file_i,\
            open(sys.argv[2], 'w', encoding='utf-8') as file_o:
        for row in filter(None, file_i.read().split('\n\n')):
            if '\t' in row:
                tokens = [x.split('\t')[0] for x in row.split('\n')]
            else:
                tokens = [x.split(' ')[0] for x in row.split('\n')]
            file_o.write(' '.join(tokens) + '\n')
