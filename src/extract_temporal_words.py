# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.

"""
Extract tense-related words candidates from Chinese conll dependency trees

"""

import sys
import zh_tree_module as tm
from collections import defaultdict



zh_file = sys.argv[1]
zh_trees = tm.treebank(zh_file)

root_error = defaultdict(int)
tmod = defaultdict(int)
advmod_ADV_RB = defaultdict(int)
aux_MD = defaultdict(int)
case_ADP_BB = defaultdict(int)
adj_aux_vDep = defaultdict(int)
noun_aux_vDep = defaultdict(int)
root_posSet=set([])
count_adj=0
count_noun=0



for doc in zh_trees:
    for tree in doc:
        if tree.root_id == 1:
            root_error[(tree.xpos_tags[1],tree.words[1])]+=1
        idx = tree.root_id
        if not idx:
            print('sentenceID=',tree.sentence_id)
            print('****no root:\n',tree)
            break

        root_dep = tree.gov2dep[idx]
        if not root_dep:
            print('sentenceID=',tree.sentence_id)
            print('****no root_dep:\n',tree)
            break

        def extract_words(idx,tree,root_dep,verb_root=True):
            edge_tmod = [x for x in root_dep if x[1]=="nmod:tmod"]
            if edge_tmod:
                for e in edge_tmod:
                    tmod[tree.words[int(e[2])]]+=1

            edge_advmod = [x for x in root_dep if x[1]=="advmod"]
            if edge_advmod:
                for e in edge_advmod:
                    if tree.upos_tags[int(e[2])] == "ADV" and tree.xpos_tags[int(e[2])]=="RB":
                        advmod_ADV_RB[tree.words[int(e[2])]]+=1

            edge_aux = [x for x in root_dep if x[1]=="aux"]
            if edge_aux:
                for e in edge_aux:
                    if tree.xpos_tags[int(e[2])]=="MD":
                        aux_MD[tree.words[int(e[2])]]+=1

            edge_patient = [x for x in root_dep if x[1]=="obl:patient"]
            if edge_patient:
                oblID = edge_patient[0][2]
                if oblID in tree.gov2dep:
                    dep_obl = tree.gov2dep[oblID]
                    edge_case = [x for x in dep_obl if x[1]=="case"]
                    if edge_case:
                        for e in edge_case:
                            if tree.upos_tags[int(e[2])] == "ADP" and tree.xpos_tags[int(e[2])]=="BB":
                                case_ADP_BB[tree.words[int(e[2])]]+=1

        if tree.upos_tags[idx]=='VERB':
            extract_words(idx,tree,root_dep)

        else:
            root_posSet.add(tree.upos_tags[idx])
            if tree.upos_tags[idx]=='NOUN':

                edge_verb = [x[1] for x in root_dep if tree.upos_tags[x[2]] in ["AUX"]]
                if edge_verb and 'cop' in edge_verb and 'aux' in edge_verb:
                    count_noun+=1
                    print(tree)

            elif tree.upos_tags[idx]=='ADJ':
                edge_verb_n = [x for x in root_dep if tree.upos_tags[x[2]] in ["AUX"]]
                if edge_verb_n:
                    count_adj+=1
                    for e in edge_verb_n:
                        id_v=e[2]
                        k = (tree.upos_tags[id_v],e[1],tree.xpos_tags[id_v])
                        adj_aux_vDep[k]+=1

