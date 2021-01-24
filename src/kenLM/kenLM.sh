#!/bin/bash

# Copyright(c) 2009 - present CNRS
# All rights reserved.

TEXT_DIR=/path/to/text
OUTPUT=/path/to/output

mkdir -p build
cd build
cmake ..
make -j 4

#training a 5-gram model with Kneser-Ney smoothing
bin/lmplz -o 5 $TEXT_DIR $OUTPUT/text.arpa

#Binarizing the model
bin/build_binary $OUTPUT/text.arpa $OUTPUT/text.bin