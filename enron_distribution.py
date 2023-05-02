import os
import importlib
os.environ["DATA_FOLDER"] = "./"
import numpy as np
import networkx as nx
import keras
from itertools import chain
from utils import datasets
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd


def parse_arff(arff_file, distribution_dictionary, is_GO=False, is_test=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                else:
                    _, f_name, f_type = l.split()
                    
                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x,i: [float(x)] if x != '?' else [np.nan])
                        
                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key:keras.utils.to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)})
                        feature_types.append(lambda x,i: d[i].get(x, [0.0]*cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                
                X.append(list(chain(*[feature_types[i](x,i) for i, x in enumerate(d_line[:len(feature_types)])])))
                
                for t in lab.split('@'): 
                    update_distribution_dict(distribution_dictionary, t)
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] =1
                    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g


def initialize_other_dataset(name, datasets, distribution_dictionary):
    is_GO, train, test = datasets[name]
    return parse_arff(train, distribution_dictionary, is_GO), parse_arff(test, distribution_dictionary, is_GO, True)

def update_distribution_dict(distribution_dictionary, t):
    label = None
    level1 = None
    level2 = None
    if len(t.split("/")) == 3:
        level1 = str(3)
    else:
        level1 = t.split("/")[0]
    
    if len(t.split("/")) >= 2:
        level2 = t.split("/")[-1]
    
    if level2 is None:
        label = level1
    else:
        label = level1 + "." + level2

    if label in distribution_dictionary:
        distribution_dictionary[label] += 1
    else:
        distribution_dictionary[label] = 1
    

distribution_dictionary = {}

initialize_other_dataset("enron_others", datasets, distribution_dictionary)

ordered_distribution_dict = OrderedDict(sorted(distribution_dictionary.items()))

plt.xticks(rotation=90)
plt.bar(ordered_distribution_dict.keys(), ordered_distribution_dict.values())
plt.show()


# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(list(ordered_distribution_dict.items()), columns=['key', 'value'])

# Write the DataFrame to an Excel file
df.to_excel('enron_distribution.xlsx', index=False)