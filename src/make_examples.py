# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.

import re
import random
import argparse
import numpy as np
import pandas as pd
import zh_tree_module as zh_tm
import fr_tree_module as fr_tm
import stanza_extract_tense as eTense
from features_SVM import *
from collections import defaultdict
import scipy.sparse as sp
import bloscpack as bp
from bert_serving.client import BertClient


parser = argparse.ArgumentParser(
    description='Dependency parsing using stanza pipeline ')

parser.add_argument('--FrTreebank', type=str, required=True,
                    help='Path to French conllu file')
parser.add_argument('--ZhTreebank', type=str, required=True,
                    help='Path to Chinese conllu file')
parser.add_argument('--output', type=str, required=True,
                    help="directory to save the datasets (tsv form)")
parser.add_argument('--featOutput', type=str, required=True,
                    help="directory to save the featSVM classifier datasets")

args = parser.parse_args()


fr_file = args.FrTreebank
zh_file = args.ZhTreebank


zh_trees = zh_tm.treebank(zh_file)
fr_trees = fr_tm.treebank(fr_file)

sent2tense = defaultdict(str)
for doc in fr_trees:
    for tree in doc:
        tense = eTense.root_tense(tree)
        if len(tense)!=1:
            print("error! more thant one major tense: ", tree.sentence_id,tense)
        sent2tense[tree.sentence_id]= tense[0]
    if len(doc)>99:
        print("error! doc length > 99")


n_docs = len(zh_trees)
docs_idx = list(range(n_docs))
random.shuffle(docs_idx)
train_len = round(n_docs*0.8)
valid_len = round(n_docs*0.1)

train_idx = docs_idx[:train_len]
valid_idx = docs_idx[train_len:train_len + valid_len]
test_idx =  docs_idx[train_len + valid_len:]
idx_UNK = [k for k,v in sent2tense.items() if v == 'UNK']

def train_dev_test(dataID,prefix="train"):
    index = []
    zh = []
    fr = []
    tense = []
    for ID in dataID:
        for i in range(1,len(zh_trees[ID])):
            zh_tree = zh_trees[ID][i]
            fr_tree = fr_trees[ID][i]
            if zh_tree.sentence_id not in idx_UNK:
                index.append(zh_tree.sentence_id)
                zh.append(''.join(zh_tree.words[1:]))
                fr.append(''.join(fr_tree.words[1:]))
                t = sent2tense[zh_tree.sentence_id]
                if t == 'PC':
                    tense.append("Past")
                else:
                    tense.append(t)
        df = pd.DataFrame({'guid':index,'label':tense,'text_zh':zh,'text_fr':fr})
        if 'test' in prefix:
            df.to_csv(args.output + prefix + ".tsv",sep='\t',index=False,header=True)
        else:
            df.to_csv(args.output + prefix + ".tsv", sep='\t', index=False, header=False)

train_dev_test(train_idx,prefix="train")
train_dev_test(valid_idx,prefix="valid")
train_dev_test(test_idx,prefix="test")
svm_test = valid_idx + test_idx
train_dev_test(svm_test,prefix="test_SVM")


# examples for SVM classifiers
wordset,WPset=make_w2idx(train_idx,zh_trees)

wordset = list(wordset)
WPset = list(WPset)
word2idx = dict(zip(wordset,range(len(wordset))))
WP2idx = dict(zip(WPset,range(len(WPset))))

n_features = len(feats_temps)+len(wordset)+len(feats_aspect)+len(feats_tmod)+len(feats_adv)+len(feats_md)+len(WP2idx)

def make_SVM_examples(dataID,n_features,prefix="train"):
    n_samples = len(dataID)
    X_matrix = np.zeros((n_samples, n_features))
    Y = []
    id_sample = 0
    for doc_idx in dataID:
        for deptree in zh_trees[doc_idx]:
            sentID = deptree.sentence_id
            if len(str(sentID)) < 3:
                sentID_str = '00' + str(sentID)
            else:
                sentID_str = str(sentID)
            # ignore the first sentence (the title) of every document
            if sentID_str[-2:] == "00":
                continue
            else:
                xfeatures, y = tree2features(deptree)
                X_matrix[id_sample] = xfeatures
                id_sample += 1
                Y.append(y)
    bp.pack_ndarray_to_file(X_matrix,args.featOutput + '/'+ prefix + "_X.blp")
    bp.pack_ndarray_to_file(np.array(Y), args.featOutput + '/' + prefix + "_Y.blp")

make_SVM_examples(train_idx,n_features)
svm_test = valid_idx + test_idx
make_SVM_examples(train_idx,n_features,prefix="test")



