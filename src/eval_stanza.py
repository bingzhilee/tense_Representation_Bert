# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.

import fr_tree_module as tm
from pud_ref_tense import ref_root_tense
from sklearn.metrics import accuracy_score
import sys

ref_conll = sys.argv[1]
stanza_conll = sys.argv[1]


ref_trees = tm.treebank(ref_conll)
stanza_trees = tm.treebank(stanza_conll)

tenses_list_ref = []
for trees in ref_trees:
    for tree in trees:
        tenses_list_ref.extend(ref_root_tense(tree))


tenses_list_stanza = []
tense_sent = []
idx = 0
c = 0
c1 = 0
for trees in stanza_trees:
    if len(trees) > 1:
        c += 1
    for tree in trees:
        if len(tree.words) < 3:
            continue  # tree as: 1  »   -   PUNCT   _   _   0   root
        tense_sent.extend(ref_root_tense(tree))
        if len(tense_sent) > 1 and 'UNK' in tense_sent:
            tense_sent.remove('UNK')
        if len(tense_sent) > 1:
            if tense_sent[0] != tense_sent[1]:
                c1 += 1
            tense_sent = [tense_sent[0]]
    tenses_list_stanza.extend(tense_sent)
    tense_sent = []
    idx += 1




accur = accuracy_score(tenses_list_ref,tenses_list_stanza)
print("Stanza predicts correctly " + str(accur) + " main tense." )











