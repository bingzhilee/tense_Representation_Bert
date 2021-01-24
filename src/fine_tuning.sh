#!/bin/bash
# Copyright(c) 2009 - present CNRS
# All rights reserved.

# You can clone the BERT's source code here : https://github.com/google-research/bert
# You can download GOOGLE's pre-trained multilingual BERT model here : \
#   https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

# path to a pre-trained model
BERT_BASE_DIR=/path/to/multi_cased_L-12_H-768_A-12
LANG="French"
# LANG="Chinese"

# path to the training data
DATA_DIR=/path/to/$LANG/dataset

mkdir -p tmp/$LANG

# path to the output diretory
OUTPUT_DIR=/path/to/tmp/$LANG

# Tense prediction task (fine-tuning)
echo "== fine-tuning =="
python run_classifier.py \
  --task_name=cola \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --do_lower_case=False \
  --save_checkpoints_steps 5000


# prediction from classifier
echo "== prediction =="
python run_classifier.py \
  --task_name=cola  \
  --do_predict=true \
  --data_dir=data/dataset\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-8721 \
  --max_seq_length=64 \
  --learning_rate=2e-5 \
  --output_dir=$OUTPUT_DIR/
