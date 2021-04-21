import pickle as pkl
import os
import numpy as np
import util_func_wo_label as uf
import networkx as nx
import matplotlib.pyplot as plt
# read the data

pathToDumpTrain = '../data/output/train/'
pathToDumpTest = '../data/output/test/'
# embed_fn = uf.embed_useT('../input/module/module_useT')
final_dic = uf.read_train(pathToDumpTrain)
first_key = list(final_dic.keys())[0]



len_feature = final_dic[first_key]['all'][0].shape[0]
result, theta = uf.calc_theta(final_dic, len_feature)
np.savetxt('theta.txt', theta)
print(theta)

embed_fn = uf.embed_useT('../input/module/module_useT')
path2data = '../data/'
# theta = np.loadtxt('theta.txt')
name_test_file = 'sample'
pair_name, all_feat_vecs = uf.all_edges_with_vec_fromfile(name_test_file, path2data, embed_fn)

all_feat_vecs0=[]
for i in range(len(all_feat_vecs)):
    if i%13==0:
        all_feat_vecs0.append(all_feat_vecs[i])

s_p = uf.S_p(theta, all_feat_vecs)

id_max = 0
max_s_p = []

for i in range(len(s_p)):
    max_s_p.append((pair_name[i][0][1], pair_name[i][1][1], s_p[i], pair_name[i][2]))

dg = nx.DiGraph()
edgeList = []

label_dic = {}
for t in max_s_p:
    edgeList.append((t[0], t[1], {"weight": t[2], "label":t[3]}))
    label_dic[(t[0], t[1])] = t[3]


# for i in range(182):
#     edgeList.append((pair_name[i][0][0], pair_name[i][1][0], {"weight": s_p_value[i], "label":pair_name[i][2]} ))
#     # Below is reversed direction of nodes.
#     #edgeList.append((pair_name[i][1][0], pair_name[i][0][0], {"weight": s_p[i]} ))
dg.add_edges_from(edgeList)
#nx.draw(dg, with_labels=True)
#plt.show()

# Add a dummy root node
dg.add_edges_from([("root", "inserted", {"weight": 0})])

maxBranch = nx.minimum_spanning_arborescence(dg, attr='weight')
# maxBranch = nx.maximum_branching(dg, attr='weight')
# edge_labels = nx.draw_networkx_edge_labels(maxBranch, pos=nx.spring_layout(maxBranch), edge_labels=label_dic)

nx.draw(maxBranch, with_labels=True)
plt.show()

sp_last = []
sp_2 = []
for i in range(len(max_s_p)):
    sp_last.append((max_s_p[i][0], max_s_p[i][1]))
    sp_2.append(max_s_p[i][2])
