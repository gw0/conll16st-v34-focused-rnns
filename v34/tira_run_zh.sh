#!/bin/bash
# Script for applying both trained parts for Chinese on TIRA
# (Discourse relation sense classifier (CoNLL16st))
#
# Usage:
#   ./tira_run_zh.sh <dataset_dir> <output_dir>
#   ./tira_run_zh.sh ./data/conll16st-zh-trial ./output
#
# Author: gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>
# License: All rights reserved

lang="zh"
dataset_dir="$1"
output_dir="$2"
model_ceq0_dir='./models-v34/conll16st-v3405-ceq0-zh-train'
config_ceq0=--config='{"filter_fn_name":"conn_eq_0", "arg1_len":500, "arg2_len":500, "words_dim":20, "focus_dim":4, "rnn_dim":20, "final_dim":100}'
model_cgt0_dir='./models-v34/conll16st-v3405-909ecb89038db38e07865a4a565b75e9'
config_cgt0=--config='{"filter_fn_name":"conn_gt_0", "arg1_len":500, "arg2_len":500, "words_dim":70, "focus_dim":5, "rnn_dim":30, "final_dim":90}'

cd "${0%/*}"
cd ..
. venv/bin/activate

echo
echo "=== running with no conn_eq_0 ==="
echo "./v34/classifier.py $lang $model_ceq0_dir $dataset_dir $output_dir $config_ceq0"
./v34/classifier.py "$lang" "$model_ceq0_dir" "$dataset_dir" "$output_dir" "$config_ceq0"

echo
echo "=== running with no conn_gt_0 ==="
echo "./v34/classifier.py $lang $model_cgt0_dir $dataset_dir $output_dir $config_cgt0"
./v34/classifier.py "$lang" "$model_cgt0_dir" "$dataset_dir" "$output_dir" "$config_cgt0"

echo
echo "Finished"
