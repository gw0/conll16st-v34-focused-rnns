#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Apply a trained discourse relation sense classifier (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import argparse
import codecs
import json
import logging
import os
import sys
from keras.models import Model, model_from_yaml
from keras.layers import Input, Embedding, TimeDistributed, Dense, merge, GRU, Dropout, Activation, Lambda, K
from keras.layers.advanced_activations import SReLU

from conll16st_data.load import Conll16stDataset
from generic_utils import debugger, load_from_pkl
from data_utils import batch_generator, decode_relation


# logging
#sys.excepthook = debugger  # attach debugger

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

# parse arguments
argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('model_dir',
    help="directory of the pre-trained model and other resources")
argp.add_argument('dataset_dir',
    help="CoNLL16st dataset directory for prediction ('parses.json', 'relations-no-senses.json')")
argp.add_argument('output_dir',
    help="output directory for system predictions ('output.json')")
argp.add_argument('--config', type=json.loads,
    help="override default model configuration (dict as string)")
args = argp.parse_args()

# experiment files
indexes_pkl = "{}/indexes.pkl".format(args.model_dir)
indexes_size_pkl = "{}/indexes_size.pkl".format(args.model_dir)
model_yaml = "{}/model.yaml".format(args.model_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.model_dir)
weights_val_hdf5 = "{}/weights_val.hdf5".format(args.model_dir)
output_json = "{}/output.json".format(args.output_dir)

# experiment configuration
log.info("configuration ({})".format(args.model_dir))

if args.config:
    config = args.config
else:
    config = {}
def c(k, d):
    log.debug("    config '{}': {} ({})".format(k, config.get(k, ""), d))
    return config.get(k, d)

epochs = int(c('epochs', 1000))  #= 200 (for real epochs)
epochs_len = int(c('epochs_len', -1))  #= -1 (for real epochs)
epochs_patience = int(c('epochs_patience', 20))  #=10 (for real epochs)
batch_size = int(c('batch_size', 64))  #= 16
snapshot_size = int(c('snapshot_size', 2048))
random_per_sample = int(c('random_per_sample', 32))

words_dim = int(c('words_dim', 20))
focus_dim = int(c('focus_dim', 4))  #3-6?
rnn_dim = int(c('rnn_dim', 20))  #10-20?
final_dim = int(c('final_dim', 100))
arg1_len = int(c('arg1_len', 100))  #= 100 (en), 500 (zh)
arg2_len = int(c('arg2_len', 100))  #= 100 (en), 500 (zh)
conn_len = int(c('conn_len', 10))  #= 10 (en, zh)
punc_len = int(c('punc_len', 2))  #=0 (en, but error), 2 (zh)
words_dropout = c('words_dropout', 0.1)  #0-0.2?
focus_dropout_W = c('focus_dropout_W', 0.33)  #0?, >0.5?
focus_dropout_U = c('focus_dropout_U', 0.66)  #0?, irrelevant?
rnn_dropout_W = c('rnn_dropout_W', 0.33)  #0.6-0.8?, irrelevant?
rnn_dropout_U = c('rnn_dropout_U', 0.33)  #0-0.5?
final_dropout = c('final_dropout', 0.5)  #0.4-0.9?, <0.5?

filter_types = None
#filter_types = ["Explicit"]
#filter_types = ["Implicit", "EntRel", "AltLex"]
filter_senses = None
#filter_senses = ["Contingency.Condition"]
filter_fn_name = c('filter_fn_name', "conn_eq_0")
if filter_fn_name == "conn_eq_0":  # connective length equals 0
    filter_fn = lambda r: len(r['Connective']['TokenList']) == 0
elif filter_fn_name == "conn_gt_0":  # connective length greater than 0
    filter_fn = lambda r: len(r['Connective']['TokenList']) > 0
else:  # no filter
    filter_fn = None

# initialize weights with pre-trained word2vec embeddings
words2vec_bin = c('words2vec_bin', None)  # en="./data/word2vec-en/GoogleNews-vectors-negative300.bin.gz"
words2vec_txt = c('words2vec_txt', None)  # zh="./data/word2vec-zh/zh-Gigaword-300.txt"

for var in ['args.model_dir', 'args.dataset_dir', 'args.output_dir', 'K._config', 'os.getenv("THEANO_FLAGS")', 'filter_types', 'filter_senses', 'filter_fn_name', 'config']:
    log.info("  {}: {}".format(var, eval(var)))

# load datasets
log.info("load dataset for prediction ({})".format(args.dataset_dir))
dataset = Conll16stDataset(args.dataset_dir, filter_types=filter_types, filter_senses=filter_senses, filter_fn=filter_fn)
log.info(dataset.summary())

# load indexes
indexes = load_from_pkl(indexes_pkl)
indexes_size = load_from_pkl(indexes_size_pkl)
log.info("  " + ", ".join([ "{}: {}".format(k, v) for k, v in indexes_size.items() ]))

# load model
#log.info("load model")
#model = model_from_yaml(open(model_yaml, 'r').read())  #XXX: broken because of Lambda layers

log.info("build model")
words2id_size = indexes_size['words2id']
rel_senses2id_size = indexes_size['rel_senses2id']
init_weights = None

shared_emb = Embedding(input_dim=words2id_size, output_dim=words_dim, weights=init_weights, dropout=words_dropout, mask_zero=True, name="shared_emb")

# input: arg1 word/token ids
arg1_ids = Input(shape=(arg1_len,), dtype='int32', name="arg1_ids")
# shape: (sample, arg1_len) of words2id_size

# input: arg2 word/token ids
arg2_ids = Input(shape=(arg2_len,), dtype='int32', name="arg2_ids")
# shape: (sample, arg2_len) of words2id_size

# input: connective word/token ids
conn_ids = Input(shape=(conn_len,), dtype='int32', name="conn_ids")
# shape: (sample, conn_len) of words2id_size

# input: punctuation word/token ids
punc_ids = Input(shape=(punc_len,), dtype='int32', name="punc_ids")
# shape: (sample, punc_len) of words2id_size

def focused_rnns(arg1_ids):
    """One RNN decides focus weights for other RNNs."""

    # embed arg1 input sequence
    arg1_emb = shared_emb(arg1_ids)
    # shape: (sample, arg1_len, words_dim)

    # focus weights for all RNNs
    focus_weights = GRU(focus_dim, return_sequences=True, dropout_U=focus_dropout_U, dropout_W=focus_dropout_W)(arg1_emb)
    # shape: (sample, arg1_len, focus_dim)

    # individual RNNs with focus
    rnns = []
    for i in range(focus_dim):
        # focus weights for current RNN
        select_repeat = Lambda(lambda x: K.repeat_elements(x[:, i], words_dim, axis=-1), output_shape=lambda s: s[:1] + (words_dim,))
        rnn_focus = TimeDistributed(select_repeat)(focus_weights)
        # shape: (samples, arg1_len, words_dim)
        # weighted input sequence
        rnn_in = merge([arg1_emb, rnn_focus], mode='mul')
        # shape: (samples, arg1_len, words_dim)

        # individual RNN
        rnn = GRU(rnn_dim, return_sequences=False, dropout_U=rnn_dropout_U, dropout_W=rnn_dropout_W)(rnn_in)
        rnns.append(rnn)
        # shape: (samples, rnn_dim)

    return rnns

# merge focused RNNs
arg1_rnns = focused_rnns(arg1_ids)
arg2_rnns = focused_rnns(arg2_ids)
conn_rnns = focused_rnns(conn_ids)
punc_rnns = focused_rnns(punc_ids)

# dense layer with logistic regression on top
x = merge(arg1_rnns + arg2_rnns + conn_rnns + punc_rnns, mode='concat')
x = Dense(final_dim)(x)
x = SReLU()(x)
x = Dropout(final_dropout)(x)
# shape: (samples, 2*hidden_dim)
x = Dense(rel_senses2id_size)(x)
x = Activation('softmax', name='rsenses')(x)
# shape: (samples, rel_senses2id_size)

inputs = [arg1_ids, arg2_ids, conn_ids, punc_ids]
outputs = [x]
losses = {
    'rsenses': c('rsenses_loss', 'categorical_crossentropy'),
}
metrics = {
    'rsenses': ['accuracy', 'loss'],
}
model = Model(input=inputs, output=outputs)

model.summary()
model.compile(optimizer=c('optimizer', "adam"), loss=losses, metrics=metrics)

# load weights
log.info("previous weights ({})".format(args.model_dir))
#model.load_weights(weights_hdf5)  # weights of best training loss
model.load_weights(weights_val_hdf5)  # weights of best validation loss

# convert from dataset to numeric format
log.info("convert from dataset ({})".format(args.dataset_dir))
x, _ = next(batch_generator(dataset, indexes, indexes_size, arg1_len, arg2_len, conn_len, punc_len, len(dataset['rel_ids']), random_per_sample=0))

# make predictions
log.info("make predictions")
y = model.predict(x, batch_size=batch_size)

# convert to CoNLL16st output format
log.info("convert predictions ({})".format(args.output_dir))
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
f_out = codecs.open(output_json, mode='a', encoding='utf8')

fallback_rel_sense = None
for rel_sense, i in indexes['rel_senses2id'].iteritems():
    if i == 2:
        fallback_rel_sense = rel_sense
none_key = None
oov_key = ""
for rel_id, y_np in zip(x['_rel_id'], y):
    rel_sense, totals = decode_relation(y_np, indexes['rel_senses2id'], indexes_size['rel_senses2id'])

    if rel_sense == none_key or rel_sense == oov_key:
        # fallback for out-of-vocabulary
        rel_sense = fallback_rel_sense
        print "fallback {} to '{}' ({})".format(rel_id, rel_sense, totals)  #XXX

    # relation output format
    rel_part = dataset['rel_parts'][rel_id]
    rel = {
        'Arg1': {'TokenList': rel_part['Arg1']},
        'Arg2': {'TokenList': rel_part['Arg2']},
        'Connective': {'TokenList': rel_part['Connective']},
        'Punctuation': {'TokenList': rel_part['Punctuation']},
        'PunctuationType': rel_part['PunctuationType'],
        'DocID': rel_part['DocID'],
        'ID': rel_id,
        'Type': 'Explicit',  # dummy, will be overwritten
        'Sense': [rel_sense],
    }
    f_out.write(json.dumps(rel) + "\n")

f_out.close()
