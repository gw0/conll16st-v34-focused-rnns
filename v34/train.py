#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Train the discourse relation sense classifier (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import argparse
import codecs
import json
import logging
import os
import sys
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, TimeDistributed, Dense, merge, GRU, Dropout, Activation, Lambda, K
from keras.layers.advanced_activations import SReLU

from generic_utils import Tee, debugger, load_from_pkl, save_to_pkl
from conll16st_data.load import Conll16stDataset
from data_utils import build_index, batch_generator, load_word2vec


# logging
sys.excepthook = debugger  # attach debugger
sys.stdout = Tee([sys.stdout])
sys.stderr = Tee([sys.stderr])

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

# parse arguments
argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('experiment_dir',
    help="directory for storing trained model and other resources")
argp.add_argument('train_dir',
    help="CoNLL15st dataset directory for training")
argp.add_argument('valid_dir',
    help="CoNLL15st dataset directory for validation")
argp.add_argument('--clean', action='store_true',
    help="clean previous experiment")
argp.add_argument('--config', type=json.loads,
    help="override default experiment configuration (dict as string)")
args = argp.parse_args()

# experiment files
console_log = "{}/console.log".format(args.experiment_dir)
indexes_pkl = "{}/indexes.pkl".format(args.experiment_dir)
indexes_size_pkl = "{}/indexes_size.pkl".format(args.experiment_dir)
train_snapshot_dir = "{}/train_snapshot".format(args.experiment_dir)
valid_snapshot_dir = "{}/valid_snapshot".format(args.experiment_dir)
model_yaml = "{}/model.yaml".format(args.experiment_dir)
model_png = "{}/model.png".format(args.experiment_dir)
metrics_csv = "{}/metrics.csv".format(args.experiment_dir)
metrics_png = "{}/metrics.png".format(args.experiment_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)
weights_val_hdf5 = "{}/weights_val.hdf5".format(args.experiment_dir)

# experiment initialization
if args.clean and os.path.isdir(args.experiment_dir):
    import shutil
    shutil.rmtree(args.experiment_dir)
if not os.path.isdir(args.experiment_dir):
    os.makedirs(args.experiment_dir)
f_log = codecs.open(console_log, mode='a', encoding='utf8')
try:
    sys.stdout.files.append(f_log)
    sys.stderr.files.append(f_log)
except AttributeError:
    f_log.close()

# experiment configuration
log.info("configuration ({})".format(args.experiment_dir))

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

for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'K._config', 'os.getenv("THEANO_FLAGS")', 'filter_types', 'config']:
    log.info("  {}: {}".format(var, eval(var)))

# load datasets
log.info("load dataset for training ({})".format(args.train_dir))
train = Conll16stDataset(args.train_dir, filter_types=filter_types, filter_senses=filter_senses, filter_fn=filter_fn)
log.info(train.summary())
if epochs_len < 0:
    epochs_len = len(train['rel_ids'])

log.info("load dataset for validation ({})".format(args.valid_dir))
valid = Conll16stDataset(args.valid_dir, filter_types=filter_types, filter_senses=filter_senses, filter_fn=filter_fn)
log.info(valid.summary())

# build indexes
if not os.path.isfile(indexes_pkl) or not os.path.isfile(indexes_size_pkl):
    log.info("build indexes")
    indexes = {}
    indexes_size = {}
else:
    log.info("previous indexes ({})".format(indexes_pkl))
    indexes = load_from_pkl(indexes_pkl)
    indexes_size = load_from_pkl(indexes_size_pkl)
indexes['words2id'], indexes_size['words2id'] = build_index(train['words'])
indexes['rel_senses2id'], indexes_size['rel_senses2id'] = build_index(train['rel_senses'])
log.info("  " + ", ".join([ "{}: {}".format(k, v) for k, v in indexes_size.items() ]))
save_to_pkl(indexes_pkl, indexes)
save_to_pkl(indexes_size_pkl, indexes_size)

init_weights = load_word2vec(indexes['words2id'], indexes_size['words2id'], words_dim, words2vec_bin, words2vec_txt)

# build model
log.info("build model")
words2id_size = indexes_size['words2id']
rel_senses2id_size = indexes_size['rel_senses2id']

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
plot(model, to_file=model_png, show_shapes=True)
with open(model_yaml, 'w') as f:
    model.to_yaml(stream=f)
model.summary()

model.compile(optimizer=c('optimizer', "adam"), loss=losses, metrics=metrics)

# initialize weights
if not os.path.isfile(weights_hdf5):
    log.info("initialize weights")
else:
    log.info("previous weights ({})".format(args.experiment_dir))
    model.load_weights(weights_hdf5)

# prepare for training
log.info("prepare snapshots")
#if not os.path.isdir(train_snapshot_dir):
#train_snapshot = next(batch_generator(train, indexes, indexes_size, arg1_len, arg2_len, conn_len, punc_len, min(len(train['rel_ids']), snapshot_size), random_per_sample=0))
#    save_dict_of_np(train_snapshot_dir, train_snapshot)
#train_snapshot = load_dict_of_np(train_snapshot_dir)
#if not os.path.isdir(valid_snapshot_dir):
valid_snapshot = next(batch_generator(valid, indexes, indexes_size, arg1_len, arg2_len, conn_len, punc_len, min(len(valid['rel_ids']), snapshot_size), random_per_sample=0))
#    save_dict_of_np(valid_snapshot_dir, valid_snapshot)
#valid_snapshot = load_dict_of_np(valid_snapshot_dir)
train_iter = batch_generator(train, indexes, indexes_size, arg1_len, arg2_len, conn_len, punc_len, batch_size, random_per_sample=random_per_sample)

# train model
log.info("train model")
callbacks = [
    ModelCheckpoint(monitor='loss', mode='min', filepath=weights_hdf5, save_best_only=True),
    ModelCheckpoint(monitor='val_loss', mode='min', filepath=weights_val_hdf5, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', patience=epochs_patience),
]
history = model.fit_generator(train_iter, nb_epoch=epochs, samples_per_epoch=epochs_len, validation_data=valid_snapshot, callbacks=callbacks, verbose=2)
log.info("training finished")

# return best result for hyperopt
results = {}
for k in history.history:
    results[k] = history.history[k][-1]  # copy others
results['loss_min'] = min(history.history['loss'])
results['acc_max'] = max(history.history['acc'])
results['val_loss_min'] = min(history.history['val_loss'])
results['val_acc_max'] = max(history.history['val_acc'])
results['epochs_len'] = len(history.history['loss'])
results['loss_'] = results['loss']
results['loss'] = -results['val_acc_max']  # objective for minimization
results['status'] = 'ok'
print("\n\n{}".format(json.dumps(results)))
