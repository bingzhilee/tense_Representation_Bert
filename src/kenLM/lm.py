#-*-coding:utf-8-*-

# Copyright(c) 2009 - present CNRS
# All rights reserved.

import argparse
import kenlm
import pandas as pd
import numpy as np
import jieba
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from statistics import mean

parser = argparse.ArgumentParser(
    description='Evaluate the performance of FineBERT in terms of the similarity between the test and train data ')

parser.add_argument('--test', type=str, required=True, help='Path to the test set (tsv format)')
parser.add_argument('--BertPredict', type=str, required=True, help='Path to the prediction file of FineBERT (csv file)')
parser.add_argument('--kenLM', type=str, required=True,help="Path to the kenLM model trained on training set")
#parser.add_argument('--zhoutput', type=str, required=True,help="directory to save the Chinese output files")
parser.add_argument('--lang', type=str, required=True,help="fr or zh")

args = parser.parse_args()

def estimate_prob(test,kenLM,lang):
    model = kenlm.LanguageModel(kenLM)
    df = pd.read_csv(test, sep='\t')
    test_text = np.array(df[str(lang)])
    test_id = np.array(df['guid'])
    proba = []
    for i in range(test_id.shape[0]):
        s_ID = test_id[i]
        sentence = test_text[i]
        if lang =="zh":
            sentence = " ".join(list(jieba.cut(sentence, cut_all=False)))
        # print(sent_tokenized)
        score = model.score(sentence)
        proba.append(score)
    # print(s_ID,score)
    df["proba"] = proba
    return df

df_proba = estimate_prob(args.test,args.kenLm)
sdf = df_proba.sort_values(by=['proba'])

pred = pd.read_csv(args.BertPredict)
gold = dict(zip(pred["sentenceID"],pred["gold"]))
predict = dict(zip(pred["sentenceID"],pred["predict"]))

def groupe_accuracy(ID_list):
    gold_test = []
    pred_test = []
    for ID in ID_list:
        gold_test.append(gold[ID])
        pred_test.append(predict[ID])
    return classification_report(gold_test,pred_test)
print(mean(sdf["proba"].tolist()[:5814]))
print(mean(sdf["proba"].tolist()[5814:11628]))
print(mean(sdf["proba"].tolist()[11628:]))

ID_list1 = sdf["guid"].tolist()[:5814]
ID_list2 = sdf["guid"].tolist()[5814:11628]
ID_list3 = sdf["guid"].tolist()[11628:]
print(groupe_accuracy(ID_list1))
print(groupe_accuracy(ID_list2))
print(groupe_accuracy(ID_list3))





