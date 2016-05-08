#!/bin/bash
# Script for applying both trained parts of a discourse relation sense classifier (CoNLL16st) on TIRA
#
# Usage:
#   ./tira_run.sh <dataset_dir> <output_dir> <model_ceq0_dir> <config_ceq0> <model_cgt0_dir> <config_cgt0>
#   ./tira_run.sh ./data/conll16st-en-trial ./output ./conll16st-v3403-ceq0 --config='{"filter_fn_name":"conn_eq_0"}' ./conll16st-v3405-cgt0  --config='{"filter_fn_name":"conn_gt_0"}'
#
# English dataset:
#   ./v34/tira_run.sh ./data/conll16st-en-trial ./tmp ./models-v34/conll16st-v3403-8c72dacfe3cce1d3f8889ccdbe197993 --config='{"filter_fn_name":"conn_eq_0", "words_dim":30, "focus_dim":4, "rnn_dim":10, "final_dim":90}' ./models-v34/conll16st-v3404-c60bcfa432d5a328876f327ef2bea88e --config='{"filter_fn_name":"conn_gt_0", "words_dim":20, "focus_dim":6, "rnn_dim":50, "final_dim":40}'
#
# Chinese dataset:
#   ./tira_run.sh ./data/conll16st-en-trial ./output 
#
# Author: gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>
# License: All rights reserved

dataset_dir="$1"
output_dir="$2"
model_ceq0_dir="$3"
config_ceq0="$4"
model_cgt0_dir="$5"
config_cgt0="$6"

cd "${0%/*}"
cd ..
. venv/bin/activate

echo
echo "=== running on conn_eq_0 ==="
echo "./v34/classifier.py $model_ceq0_dir $dataset_dir $output_dir $config_ceq0"
./v34/classifier.py "$model_ceq0_dir" "$dataset_dir" "$output_dir" "$config_ceq0"

echo
echo "=== running on conn_gt_0 ==="
echo "./v34/classifier.py $model_cgt0_dir $dataset_dir $output_dir $config_cgt0"
./v34/classifier.py "$model_cgt0_dir" "$dataset_dir" "$output_dir" "$config_cgt0"

echo
echo "Finished"
