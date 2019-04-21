"""
Convert output of Tagger to CoNLL format.
"""
import sys

if __name__ == '__main__':
    with open(sys.argv[1]) as file_p,\
            open(sys.argv[2], 'w') as file_o:
        for line in filter(None, file_p.read().split('\n')):
            tokens_tags = line.split(' ')
            tokens = []
            tags = []
            for tok_tag in tokens_tags:
                token, tag = tok_tag.split('__')
                tokens.append(token)
                tags.append(tag)
            for tag in tags:
                file_o.write(tag + '\n')
            file_o.write('\n')
