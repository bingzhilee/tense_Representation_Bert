# Copyright(c) 2009 - present CNRS
# All rights reserved.

BERT_BASE_DIR=/path/to/multi_cased_L-12_H-768_A-12
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`

bert-serving-start -model_dir $BERT_BASE_DIR -num_worker=10
