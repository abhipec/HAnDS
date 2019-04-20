"""
Plot the pie charts to show type subsumption.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 12.0
    labels = 'PER', 'LOC', 'ORG', 'MISC', 'Out of\nscope'
    sizes_figer = [16, 21, 19, 30, 14]
    sizes_typenet = [22, 18, 17, 18, 25]
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
    explode = (0, 0, 0, 0, 0.1)
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(9, 4))
    ax1.pie(sizes_figer, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=70, labeldistance=1.11, colors=colors)
    ax2.pie(sizes_typenet, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=70, labeldistance=1.11, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_xlabel('Distribution of 113 entity types\nof the FIGER type set.')
    ax2.set_xlabel('Distribution of 1081 entity types\nof the TypeNet type set.')
    plt.tight_layout() 
    plt.savefig('conll_subsumtion.svg')
