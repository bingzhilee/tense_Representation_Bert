# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.
import argparse
from bert_serving.client import BertClient
import bloscpack as bp
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(
    description='extract sentences representations from pre-trained BERT model ')

parser.add_argument('--input', type=str, required=True, help='Path to the dataset (tsv format)')
parser.add_argument('--output', type=str, required=True,help="directory to save the French or English BERT output files")
#parser.add_argument('--zhoutput', type=str, required=True,help="directory to save the Chinese output files")
parser.add_argument('--lang', type=str, required=True,help="fr or zh")
parser.add_argument('--prefix', type=str, required=True,help="train or test")

args = parser.parse_args()


bc = BertClient(check_length=False)

def bert_sent_representation(input,output, lang,prefix):
    if lang == "fr":
        bert_service(input,output,prefix,3)
    if lang == "zh":
        bert_service(input, output, prefix, 2)

def bert_service(input,output,lang, prefix,n_col):
    df = pd.read_csv(input, delimiter='\t', usecols=[1, n_col], header=None)
    df.columns = ['tense', 'text_'+lang]
    X_list = df['text_'+lang].values.tolist()
    Y = np.array(df['tense'])
    list_vec = bc.encode(X_list)
    bp.pack_ndarray_to_file(list_vec, output + '/' + prefix + "_X.blp")
    bp.pack_ndarray_to_file(Y, output + '/'+ prefix + "_Y.blp")


bert_sent_representation(args.input,args.output,args.lang,args.prefix)