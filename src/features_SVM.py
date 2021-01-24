# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.

import numpy as np


# tense-related features
feats_md = ['可能', '能', '会', '未能', '必须', '可以', '应当', '足以', '不会', '不能', '需要', '应该', '应', '能否', \
            '能够', '要', '想', '可', '不可能', '才能', '不该', '不必', '尽可能', '不难', '并不会', '只能', '不愿', '不可',\
            '不要', '愿意', '需', '未必', '不得不', '请', '不只', '不想','得','必需','必','必定','必将','不曾']
md2idx = dict(zip(feats_md,range(len(feats_md))))

feats_tmod = ['事实上','实际上','如今','现','现在','现今','目前','当前','眼下','同时','同期','与此同时',\
              '前','从前','此前','之前','以前','不久前','最终','最后','今天','当今','今年','迄今','时至今日','今时','今日',\
              '明天','明年','后天','后年','昨天','去年','每年','每天','每月','每个星期','当时','当年','那时','当初','最初','过去','时',\
              '间','时候','期间','时期','后','其后','之后','事后','今后','此后','以后','战后','来','近年来','近几年来','近几年',\
              '近期','最近','长期以来','后来','未来','届时','永远','不久','此时','此刻','自此','本周','本月','以往', '假以时日','上个月','下个月','这次','上次','下次','刚才']
tmod2idx = dict(zip(feats_tmod,range(len(feats_tmod))))

feats_adv = ['已经','因此','正在','一直','在','仍然','仍','已','也许','正','依然','未','再','曾','往往','这就','不再','通常','常常','曾经','诚然',\
             '很快','不断','经常','日益','总是','很少','没','从来','早已','必然','即将','永远','尚未','刚刚','绝不','多年来','纷纷','至今','一般',\
             '立刻','依旧','实际上''尚','早','毫不','一再','立即','随之','马上','原本','就要','从未','总','而后','始终','本来','常','暂时','必将',\
             '后','最近','向来','近','时常','多年''众所周知','大都','从不','将要','快','终将','从此','也曾','未曾' ]
advmod2idx = dict(zip(feats_adv,range(len(feats_adv))))

feats_temps = ['UNK','Past','Pres','Fut']
temps2idx = dict(zip(feats_temps,range(len(feats_temps))))

feats_aspect = ['了','过', '着']
mark2idx = dict(zip(feats_aspect,range(len(feats_aspect))))
valence_mark = ['将','把','被']

def make_w2idx(train_idx,zh_trees):
    wordset = set([])
    WPset = set([]) # word and POS tag combination
    punct = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    for doc_idx in train_idx:
        for sent_tree in zh_trees[doc_idx]:
            len_s = len(sent_tree.words)
            for node in range(1, len_s):
                word = sent_tree.words[node]
                xpos = sent_tree.xpos_tags[node]
                upos = sent_tree.upos_tags[node]
                if upos != "PUNCT" and (not bool(re.search(punct, word))) and (
                not bool(re.search('[a-zA-Z]', word))):  # xpos != 'FW': #
                    wordset.add(word)
                    if xpos == "FW":
                        print(word)
                    WPset.add((word, xpos))
    return wordset, WPset



def oneHotEncoding(input, w2id_dico):
    if input:

        feats = np.zeros(len(w2id_dico))
        for word in input:
            if word in w2id_dico:  # manages unk words (ignore)
                feats[w2id_dico[word]] = 1
    else:
        feats = np.zeros(len(w2id_dico))
    return feats


def tree2features(deptree,sent2tense,word2idx,WP2idx):
        # feature n°1 context tense: the major tense of the precedent sentence
        sentID = deptree.sentence_id
        # we don't analyse the 1st sentence (title) of each doc, so the contexte of the 1st sent after title is 'UNK'
        context_tense = [sent2tense[sentID - 1]]
        ylabel = temps2idx[sent2tense[sentID]]
        F_context = oneHotEncoding(context_tense, temps2idx)
        aspect_mark = set([])
        tmod = set([])
        advmod = set([])
        md = set([])
        WP = set([])

        root_id = deptree.root_id
        if not root_id:
            print('sentenceID=', deptree.sentence_id)
            print('****no root:\n', deptree)
            print('**** words: ', deptree.words)
            print('**** xpos: ', deptree.xpos_tags)
            F_VV = oneHotEncoding([deptree.words[1]], word2idx)
            F_aspect = oneHotEncoding(aspect_mark, mark2idx)
            F_tmod = oneHotEncoding(tmod, tmod2idx)
            F_advmod = oneHotEncoding(advmod, advmod2idx)
            F_md = oneHotEncoding(md, md2idx)
            F_WP = oneHotEncoding([(deptree.words[1], deptree.xpos_tags[1])], WP2idx)

        else:
            root_dep = deptree.gov2dep[root_id]  # pas sûr d'avoir toujours root-dep?
            root_deprel = [edge[1] for edge in root_dep]

            # feature n°2: major verb VV of a sentence
            # -if root is a verb or if non-verb root don't have AUX dependent, then VV==root_word
            # -if root is not a verb and AUX dependent exist,
            # if AUX aux (/MD) exist, VV = premier MD_word; elif AUX cop exist, VV=cop_word;
            # else VV=root_word
            if deptree.upos_tags[root_id] == "VERB":
                VV_id = root_id
            elif "aux" in root_deprel:
                edge_aux = [x for x in root_dep if x[1] == "aux" and deptree.upos_tags[x[2]] == "AUX"]
                if edge_aux:
                    VV_id = edge_aux[0][2]
                else:
                    VV_id = root_id
            elif "cop" in root_deprel and deptree.upos_tags[root_id] != "ADJ":
                edge_cop = [x for x in root_dep if x[1] == "cop" and deptree.upos_tags[x[2]] == "AUX"]
                if edge_cop:
                    VV_id = edge_cop[0][2]
                else:
                    VV_id = root_id
            else:
                VV_id = root_id
            VV = [deptree.words[VV_id]]
            F_VV = oneHotEncoding(VV, word2idx)

            # feature n°3 aspect marker
            if 'case:aspect' in root_deprel:
                aspect_edge = [x for x in root_dep if x[1] == "case:aspect"]
                aspect_mark.add(deptree.words[aspect_edge[0][2]])
            F_aspect = oneHotEncoding(aspect_mark, mark2idx)

            # feature n°4 nominal tense modifier
            if "nmod:tmod" in root_deprel:
                edge_tmod = [x for x in root_dep if x[1] == "nmod:tmod"]
                for edge in edge_tmod:
                    if deptree.words[edge[2]] in feats_tmod:
                        tmod.add(deptree.words[edge[2]])
            F_tmod = oneHotEncoding(tmod, tmod2idx)

            # feature n°5 nominal tense modifier
            if "advmod" in root_deprel:
                edge_advmod = [x for x in root_dep if x[1] == "advmod"]
                for edge in edge_advmod:
                    if deptree.words[edge[2]] in feats_adv:
                        advmod.add(deptree.words[edge[2]])
            F_advmod = oneHotEncoding(advmod, advmod2idx)

            # feature n°6 modal auxiliaire
            if "aux" in root_deprel:
                edge_aux = [x for x in root_dep if x[1] == "aux" and deptree.xpos_tags[x[2]] == "MD"]
                if edge_aux:
                    for edge in edge_aux:
                        if deptree.words[edge[2]] in feats_md:
                            md.add(deptree.words[edge[2]])
            F_md = oneHotEncoding(md, md2idx)

            # feature n°7 word/POS
            for node in range(1, len(deptree.words)):
                WP.add((deptree.words[node], deptree.xpos_tags[node]))
            F_WP = oneHotEncoding(WP, WP2idx)


        x1 = np.concatenate((F_context,F_aspect,F_VV))
        x2 = np.concatenate((x1,F_tmod,F_advmod))
        xfeatures = np.concatenate((x2,F_md,F_WP))

        return xfeatures.reshape(1,-1),ylabel

