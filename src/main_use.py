import numpy as np
import json
import os
import pandas as pd
import util_func as uf
import string
import nltk.data
import tensorflow_hub as hub
import pickle as pkl

embed_fn = uf.embed_useT('../input/module/module_useT')

path2data = '../data/'
path2dump = '../data/output/train/'
list_files = os.listdir(path2data + 'list')
flow_files = os.listdir(path2data + 'flow')


with open(path2data + 'r-100.json') as f:
    recipe_title_list = json.loads(f.read())

# recipe_title_list = ['sample', 'test']
num_recipes = len(recipe_title_list)

for recp_id in range(num_recipes):

    current_rec_title = recipe_title_list[recp_id]['title']
    # current_rec_title = recipe_title_list[recp_id]
    print(current_rec_title)
    Master = uf.make_feature_vector_dict(current_rec_title, path2data, embed_fn)
    # dump = {}
    # dump['f_train'] = training_pair_vec
    # dump['f_all'] = all_pairs_vec
    # dump['pair_name'] = all_pairs

    f1 = open(path2dump + str(current_rec_title) + '.out', 'wb')
    pkl.dump(Master, f1, protocol=2)
    f1.close()
# minimize the loss to find theta

# theta0 = np.random.rand(len_feature)
# theta = theta0/np.sum(theta0)
# result, theta = uf.opt_func(theta, training_pair_vec, all_pairs_vec)

# s_p = uf.S_p(theta, all_pairs_vec)


# np.savetxt('sp.txt', s_p)
# np.savetxt('link.txt', id_link)
# print(id_link)
