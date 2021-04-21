import numpy as np
import json
import os
import pandas as pd
from nltk.corpus import stopwords
import string
from scipy.optimize import minimize
import nltk
import pickle as pklc
import tensorflow as tf
import tensorflow_hub as hub
import pickle as pkl

def read_list_flow(file_name, path2data):
    rec_n = file_name.replace("'", "")
    rec_n = rec_n.replace("-", "")
    list_path = path2data + 'list/' + rec_n + '.list'
    flow_path = path2data + 'flow/' + rec_n + '.flow'

    with open(list_path, 'r') as f:
            list_0 = []
            for ele in f:
                line = ele.split('\n')
                line = [words for segments in line for words in segments.split()]
                list_0.append(line)

    with open(flow_path, 'r') as f:
            flow_0 = []
            for ele in f:
                line = ele.split('\n')
                line = [words for segments in line for words in segments.split()]
                flow_0.append(line)
    return list_0, flow_0

def sep_list(list_0):
    list_1 = np.asarray(list_0)
    l_word = list_1[:,3]
    l_ID_0 = list_1[:,0:3]
    l_ID = [l_ID_0[i][0] + "-" + l_ID_0[i][1] + "-" + l_ID_0[i][2] for i in range(len(l_ID_0))]
    l_ner_cod = list_1[:,4]
    l_ner_gt = list_1[:,5]
    return l_word, l_ID, l_ner_cod, l_ner_gt

def sep_flow(flow_0):
    flow_1 = np.asarray(flow_0)
    f_ID_in0 = flow_1[:,0:3]
    f_ID_in = [f_ID_in0[i][0] + "-" + f_ID_in0[i][1] + "-" + f_ID_in0[i][2] for i in range(len(f_ID_in0))]

    f_ID_out0 = flow_1[:,4:8]
    f_ID_out = [f_ID_out0[i][0] + "-" + f_ID_out0[i][1] + "-" + f_ID_out0[i][2] for i in range(len(f_ID_out0))]
    f_link = flow_1[:,3]
    return f_ID_in, f_ID_out, f_link

# using gensim
def featureVecMethod(words, model, num_features):
    # Function to average all word vectors in a paragraph
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0

    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)

    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec.reshape(1, num_features)

def getAvgFeatureVecs(reviews, model, num_features):
    # Function for calculating the average feature vector
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1

    return reviewFeatureVecs

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

# def find_nonsw_p(l_word):
#     id_non_sw_p = []
#     punctuations = list(string.punctuation)
#     a = punctuations + stopwords.words()
#     #find the index of the words which are not punctuation or stop-word
#     for num_word in range(len(l_word)):
#         if l_word[num_word] not in  a:
#                 id_non_sw_p.append(num_word)
#     return id_non_sw_p

# Given a lists of f_id_in, f_id_out, f_link, and a two id's A and B
# Return the link label on the link that connects A to B. If no link exists return None.
def get_link_tag(inList, outList, linkList, idIn, idOut):

    # Look through the inList until you find idIn in the inList.
    for i in range(len(inList)):
        if inList[i] == idIn and outList[i] == idOut:
            return linkList[i]

    # At this point no link between idIn and idOut were found
    return None



def S_p(theta, all_pairs_vec):
    all_pairs_vec = np.asarray(all_pairs_vec)
    dot_all = np.exp(np.dot(theta, all_pairs_vec.T))
    S_p = dot_all / np.sum(dot_all)
    return S_p


def loss_func(theta, final_dic):
    number = len(final_dic)
    loss_sum = 0
    for ID in final_dic:
        train = final_dic[str(ID)]['training']
        train = np.asarray(train)
        all_data = final_dic[str(ID)]['all']
        all_data = np.asarray(all_data)
        if len(train)!=0:
            dot_train = np.sum(np.dot(theta, train.T))
        else:
            dot_train = 0
        
        dot_all = np.sum(np.exp(np.dot(theta, all_data.T)))
        loss_sum = -(dot_train - np.log(dot_all)) + loss_sum 
        
    loss = loss_sum + .5 *np.dot(theta, theta.T)
    # loss =  -np.sum(np.log(dot_num/np.sum(dot_denum))) + .5 *np.dot(theta, theta.T)
    return loss
# def loss_func(theta, dot_t, dot_all):
#     loss = - np.sum(dot_t)/np.sum(dot_all) + .5 *np.dot(theta, theta.T)
#     return loss

def opt_func(theta, final_dic):
    result = minimize(loss_func, theta, args=(final_dic),
                  method='Nelder-Mead', tol=1e-3)
    theta = result.x
    return result, theta

# """
# # Test get_link_tag. Looking at the file Almond_and_apple_cake_recipe check
# # a few links
# list_0, flow_0 = read_list_flow("Almond_and_apple_cake_recipe__All_recipes_UK",
#                                 "../data/")
# f_ID_in, f_ID_out, f_link = sep_flow(flow_0)
# print(get_link_tag( f_ID_in, f_ID_out, f_link, "1-1-13", "1-1-1"))
# print(get_link_tag( f_ID_in, f_ID_out, f_link, "5-1-76", "6-1-1"))
# """

# Given a file name and path return a list of pairs.
# Each pair is a list of 2 tuples, a tuple for each node.
# Within the tuple is the id, word, r-NER, POS
# i.e. (1-2-26, measurement, NN1, St-I)
def all_pairs_from_list_file(file_name, path2data):

    list_0, flow_0 = read_list_flow(file_name, path2data)
    l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)
    punctuations = list(string.punctuation)
    # addTopunc = ['a', 'and', 'of', 'to', 'mightn', "mightn't", "mustn", "mustn't", "needn't", "shan't"]
    # punctuations = punctuations + addTopunc
    l_word_nop = []
    all_pairs = []
    for i in range(len(l_word)):
        if l_word[i] not in punctuations:
            l_word_nop.append(l_word[i])
            for j in range(len(l_word)):
                if l_word[j] not in punctuations:
                    if i != j:
                        p1 = (l_ID[i], l_word[i], l_ner_cod[i], l_ner_gt[i])
                        p2 = (l_ID[j], l_word[j], l_ner_cod[j], l_ner_gt[j])
                        all_pairs.append([p1, p2])

    return all_pairs, l_word_nop

# Test the all_pairs_from_list_file()
# allPs = all_pairs_from_list_file("test", "../data/")
# for p in allPs:
#     print(p)

# Given a pair of words (one row from the all_pairs list)
# return a np array indicating if in same, previous, or subsequent
# sentence. The array is 1 x 3. index 0 means same, 1 previous, 2, subsequent
def get_in_same_sent_feature( pair ):

    # Subtract the sentence values
    id1 = pair[0][0]
    step1 = int(id1.split("-")[0])
    sent1 = int(id1.split("-")[1])
    id2 = pair[1][0]
    step2 = int(id2.split("-")[0])
    sent2 = int(id2.split("-")[1])

    stepDiff = step1 - step2
    returnVector = np.zeros(3)
    if stepDiff < 0:    # s1 is before s2
        return np.asarray([0,1,0])
    elif stepDiff > 0:
        return np.asarray([0,0,1])

    # At this point the two words are in the same step.
    diff = sent1 - sent2
    returnVector = np.zeros(3)
    if diff < 0:    # s1 is before s2
        returnVector = np.asarray([0,1,0])
    elif diff == 0:
        returnVector = np.asarray([1,0,0])
    else:
        returnVector = np.asarray([0,0,1])

    return returnVector


# Test get_in_same_sent_feature()
# p1 = [("1-1-1", "Preheat", "VV0", "Ac-B" ), ("2-1-9", "the", "VV0", "Ac-B" )]
# print(get_in_same_sent_feature(p1))


# Given a pair of words and the sorted list of ids in the .list file
# count the words between the words in the pair.
def count_words_between(pair, list_of_ids):
    idU = pair[0][0]
    idV = pair[1][0]

    index = 0
    while index < len(list_of_ids):         # Find the starting point
        if list_of_ids[index] == idU:
            multiplier = 1
            stopId = idV
            break
        if list_of_ids[index] == idV:
            multiplier = -1
            stopId = idU
            break
        index += 1

    count = 0                           # Count until reach the other id.
    while index < len(list_of_ids) and stopId != list_of_ids[index]:
        index += 1
        count += 1

    return count * multiplier

# """
# # Test count_words_between
# list_0, flow_0 = read_list_flow("Almond_and_apple_cake_recipe__All_recipes_UK",
#                                 "../data/")
# l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)
# p1 = [("1-1-9", "the", "VV0", "Ac-B" ), ("6-2-12", "to", "VV0", "Ac-B" )]
# print(count_words_between(p1, l_ID))
# """
# Returns True is given word is a preposition and false
# otherwise. Preposition is determined by NLTK pos_tag() function
# Tags 'RP', 'IN', and 'TO' are considered prepositions.
def is_preposition(word):
    tag = nltk.pos_tag([word])[0][1]
    if tag == 'RP' or tag == 'IN' or tag == 'TO':
        return True
    else:
        return False

# Given a pair of words and the ordered wordList, and the listOfIds from the recipe
# return True if the pair of words is in the same sentence and a
# preposition lies between them.
def is_same_sent_with_prep( pair, listOfIds, wordList ):

    # First check if it is in the same sentence
    vec = get_in_same_sent_feature( pair )
    if vec[0] != 1:
        return np.array([0])

    # Count how many words are between the pair
    numWordsBetween = count_words_between(pair, listOfIds)

    # Find the starting index
    if numWordsBetween < 0:
        startId = pair[1][0]
    else:
        startId = pair[0][0]
    index = 0
    while listOfIds[index] != startId:
        index += 1

    # Now index is the starting index
    index += 1 # Go beyond the starting word into the between words
    numWordsBetween = abs(numWordsBetween)
    for i in range(index, index + numWordsBetween):
        #print(wordList[i])
        #print(is_preposition(wordList[i]))
        if is_preposition(wordList[i]):
            return np.array([1])

    return np.array([0])



# # Test is_same_sent_with_prep
# list_0, flow_0 = read_list_flow("Almond_and_apple_cake_recipe__All_recipes_UK",
#                                 "../data/")
# l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)
# p1 = [("1-1-1", "to", "VV0", "Ac-B" ), ("1-1-40", "Preheat", "VV0", "Ac-B" )]
# print(is_same_sent_with_prep( p1, l_ID, l_word ))

def all_pairs_from_list_file(file_name, path2data):

    list_0, flow_0 = read_list_flow(file_name, path2data)
    l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)

    tupleList = []
    # Starting at the bottom ignore all o's and combine the I's
    # As you go create a list of tuples.
    i = len(l_word) - 1
    while i >= 0:
        # print(i)
        if l_ner_gt[i] == 'O':
            i -= 1
        else:
            tag, bio = l_ner_gt[i].split("-")
            if bio == 'B':
                tupleList.append( (l_ID[i], l_word[i], l_ner_cod[i], tag) )
            else:
                word = l_word[i]
                i -= 1
                while l_ner_gt[i].split("-")[1] == 'I':     # Keep going until you reach a 'B'
                    word = l_word[i] + "_" + word
                    i -= 1
                tupleList.append( (l_ID[i], l_word[i] + "_" + word, l_ner_cod[i], tag) )
            i -= 1
    # Make a list of all combinations of pairs of tuples.
    all_pairs = []
    for i in range(len(tupleList)):
        for j in range(len(tupleList)):
            if i != j:
                all_pairs.append([tupleList[i], tupleList[j]])

    return all_pairs


def read_train(pathToDumpFiles):
    all_recip = os.listdir(pathToDumpFiles)
    counter = 0
    final_dic = {}
    for i in all_recip:
        f = open(pathToDumpFiles + i, "rb")
        data = pkl.load(f)
        f.close()
        final_dic.update(data)

    return final_dic

# def read_test(pathToDumpTest, theta):

#     f = open(pathToDumpTest, "rb")
#     data = pkl.load(f)

#     featVec_all = data['f_all']
#     pair_name = data['pair_name']
#     s_p = S_p(theta, featVec_all)

#     return pair_name, s_p

def calc_theta(final_dic, len_feature):
    theta0 = np.random.rand(len_feature)
    theta = theta0/np.sum(theta0)
    result, theta = opt_func(theta, final_dic)
    return result, theta

def all_triples_from_list_file(file_name, path2data):

    list_0, flow_0 = read_list_flow(file_name, path2data)
    l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)

    tupleList = []
    # Starting at the bottom ignore all o's and combine the I's
    # As you go create a list of tuples.
    i = len(l_word) - 1
    while i >= 0:
        # print(i)
        if l_ner_gt[i] == 'O':
            i -= 1
        else:
            tag, bio = l_ner_gt[i].split("-")
            if bio == 'B':
                tupleList.append( (l_ID[i], l_word[i], l_ner_cod[i], tag) )
            else:
                word = l_word[i]
                i -= 1
                while l_ner_gt[i].split("-")[1] == 'I':     # Keep going until you reach a 'B'
                    word = l_word[i] + "_" + word
                    i -= 1
                tupleList.append( (l_ID[i], l_word[i] + "_" + word, l_ner_cod[i], tag) )
            i -= 1
    # Make a list of all combinations of pairs of tuples.
    all_triples = []
    for i in range(len(tupleList)):
        for j in range(len(tupleList)):
            if i != j:
                for tag in ['Agent', 'Targ', 'Dest', 't-comp', 'F-comp', 'F-eq', 'F-part-of','F-set', 'T-eq', 'T-part-of', 'A-eq', 'V-tm', 'other-mod']:
                    all_triples.append([tupleList[i], tupleList[j], tag])

    return all_triples

def get_feature_vec_from_vertices(v1, v2, l_ID, l_word, edge_label, embed_fn):
    
    first_word = v1[1]
    sec_word = v2[2]
    step_pair0 = first_word + ' ' + sec_word
    step_pair1 = step_pair0.split('_')
    step_pair = [' '.join(step_pair1)]

    # find the id of words in the pair
    idIn = v1[0]
    idOut = v2[0]
    #id_link.append((idIn,idOut))

    # feature 1: find the word embedding

    f_word = embed_fn(step_pair)
    len_f_word = f_word.shape[1]


    # feature 2: get_in_same_sent_feature
    f_sentence = get_in_same_sent_feature([v1, v2])

    # feature 3: count_words_between
    f_words_btw = np.array([count_words_between([v1, v2], l_ID)])
    f_words_btw = f_words_btw/np.abs(np.sum(f_words_btw))
    # feature 4: is_same_sent_with_prep( pair, listOfIds, wordList )
    f_prep = is_same_sent_with_prep([v1, v2], l_ID, l_word )

    f_tag = edge_label_to_vector(edge_label)

    # f_all = (f_word.reshape(len_f_word), f_sentence, f_prep, f_tag)
    f_all = (f_word.reshape(len_f_word), f_sentence, f_words_btw, f_prep, f_tag)

    f_vec = np.concatenate(f_all)
    return f_vec

"""
# Test the get_feature_vec_from_vertices()
list_0, flow_0 = read_list_flow("test",
                                 "../data/")
l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)
v1 = ("1-1-1", "Preheat", "VV0", "Ac-B" )
v2 = ("1-1-13", "oven", "NN1", "T-B" )
print(get_feature_vec_from_vertices( v1, v2, l_ID, l_word ))
"""

# Return a vector representing the edge labels.
def edge_label_to_vector(label):
    myDict = {
    'a':     np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    't':      np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'd':      np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    't-comp':    np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'f-comp':    np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    'f-eq':      np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    'f-part-of': np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'f-set':     np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    't-eq':      np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    't-part-of': np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    'a-eq':      np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    'v-tm':      np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    'o': np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    return myDict[label]


def make_feature_vector_dict(file_name, path2data, embed_fn):
    list_0, flow_0 = read_list_flow(file_name, path2data)
    f_ID_origin, f_ID_destination, f_link = sep_flow(flow_0)
    l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)

    tupleList = []
    # Starting at the bottom ignore all o's and combine the I's
    # As you go create a list of tuples.
    i = len(l_word) - 1
    while i >= 0:
        # print(i)
        if l_ner_gt[i] == 'O':
            i -= 1
        else:
            tag, bio = l_ner_gt[i].split("-")
            if bio == 'B':
                tupleList.append( (l_ID[i], l_word[i], l_ner_cod[i], tag) )
            else:
                word = l_word[i]
                i -= 1
                while l_ner_gt[i].split("-")[1] == 'I':     # Keep going until you reach a 'B'
                    word = l_word[i] + "_" + word
                    i -= 1
                tupleList.append( (l_ID[i], l_word[i] + "_" + word, l_ner_cod[i], tag) )
            i -= 1

    # Now that we have a list of all vertices in the training data. Let's make feature vectors
    # of the edges in the training data and the possible edges.
    # Initialize the master dictionary
    master = {}
    print("master type" + str(type(master)))
    for i in range(len(tupleList)):
        u_id = tupleList[i][0] + str(file_name)
        master[u_id] = {}
        print("master[" + u_id + "] type" + str(type(master[u_id])))
        for j in range(len(tupleList)):
            if i != j:
                v_id = tupleList[j][0]
                master[u_id]["training"] = []
                master[u_id]["all"] = []
                idIn = tupleList[i][0]
                idOut = tupleList[j][0]
                edge_label = get_link_tag(f_ID_origin, f_ID_destination, f_link, idIn, idOut)
                for tag in ['a', 't', 'd', 't-comp', 'f-comp', 'f-eq', 'f-part-of','f-set', 't-eq', 't-part-of', 'a-eq', 'v-tm', 'o']:
                    feat_vec = get_feature_vec_from_vertices(tupleList[i], tupleList[j], l_ID, l_word, tag, embed_fn)
                    master[u_id]["all"].append(feat_vec)
                    # If this combination of u,v, and tag is in the training data add to training list
                    if edge_label and edge_label == tag:
                        master[u_id]["training"].append(feat_vec)
    return master

# # # Test the make_feature_vector_dict()
# master = make_feature_vector_dict("small_test", "../data/")
# print(master["1-1-1"]["all"])
# print(master["1-1-1"]["training"])
# print(master["1-1-13"]["training"])
# print(master["1-1-13"]["all"])
# #print(master)
# #for m in master:
# #    print(m)
# """
# print("Vertex: " + m)
# print("\tTraining")
# for t in m["training"][0]:
#     print("\t\t" + t)
# for a in m["all"][0]:
#     print("\t\t" + a)
# # """
# for p in allPs:
#     print(p)
# print(len(allPs))

def all_edges_with_vec_fromfile(file_name, path2data, embed_fn):

    list_0, flow_0 = read_list_flow(file_name, path2data)
    l_word, l_ID, l_ner_cod, l_ner_gt = sep_list(list_0)

    tupleList = []
    # Starting at the bottom ignore all o's and combine the I's
    # As you go create a list of tuples.
    i = len(l_word) - 1
    while i >= 0:
        # print(i)
        if l_ner_gt[i] == 'O':
            i -= 1
        else:
            tag, bio = l_ner_gt[i].split("-")
            if bio == 'B':
                tupleList.append( (l_ID[i], l_word[i], l_ner_cod[i], tag) )
            else:
                word = l_word[i]
                i -= 1
                while l_ner_gt[i].split("-")[1] == 'I':     # Keep going until you reach a 'B'
                    word = l_word[i] + "_" + word
                    i -= 1
                tupleList.append( (l_ID[i], l_word[i] + "_" + word, l_ner_cod[i], tag) )
            i -= 1
    # Make a list of all combinations of pairs of tuples.
    all_triples = []
    all_feat_vecs = []
    for i in range(len(tupleList)):
        for j in range(len(tupleList)):
            if i != j:
                for tag in ['a', 't', 'd', 't-comp', 'f-comp', 'f-eq', 'f-part-of','f-set', 't-eq', 't-part-of', 'a-eq', 'v-tm', 'o']:
                    all_triples.append([tupleList[i], tupleList[j], tag])
                    all_feat_vecs.append(get_feature_vec_from_vertices(tupleList[i], tupleList[j], l_ID, l_word, tag, embed_fn))

    return all_triples, all_feat_vecs


def countAllLinkTypes():

    # Get all the files in the flow folder.
    pathToData = "../data/"
    listOfFlowFiles = os.listdir(pathToData + 'flow/')
    print("Number of files in flow directory: " + str(len(listOfFlowFiles)))

    listOfTypes = ['a', 't', 'd', 't-comp', 'f-comp', 'f-eq', 'f-part-of','f-set', 't-eq', 't-part-of', 'a-eq', 'v-tm', 'o', 'v', 's','-']
    # Initialize a dictionary to hold all the counts
    countD = {}
    for k in listOfTypes:
        countD[k] = 0
    # """
    # file = "test"
    # list_0, flow_0 = read_list_flow("test", "../data/")
    # f_ID_in, f_ID_out, f_link = sep_flow(flow_0)
    # for l in f_link:
    #     countD[l] += 1
    # """

    for file in listOfFlowFiles:
        list_0, flow_0 = read_list_flow(file[:-5], pathToData)
        f_ID_in, f_ID_out, f_link = sep_flow(flow_0)
        for l in f_link:
            countD[l] += 1

    for k in countD:
        print(k + ": " + str(countD[k]))