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


def build_index(sequences, max_new=None, min_count=1, index=None, vocab_start=2, none_key=None, none_ids=0, oov_key="", oov_ids=1):
    """Build vocabulary index from dicts or lists of strings (reserved ids: 0 = none/padding, 1 = out-of-vocabulary)."""
    if index is None:
        index = {}

    def _traverse_cnt(obj, cnts):
        """Recursively traverse dicts and lists of strings."""
        if isinstance(obj, dict):
            for s in obj.itervalues():
                _traverse_cnt(s, cnts)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            for s in obj:
                _traverse_cnt(s, cnts)
        else:
            try:
                cnts[obj] += 1
            except KeyError:
                cnts[obj] = 1

    # count string occurrences
    cnts = {}
    _traverse_cnt(sequences, cnts)

    # ignore strings with low occurrences
    for k, cnt in cnts.items():
        if cnt < min_count:
            del cnts[k]

    # rank strings by decreasing occurrences and use as index
    index_rev = sorted(cnts, key=cnts.get, reverse=True)
    if max_new is not None:
        index_rev = index_rev[:max_new]  # limit amount of added strings

    # mapping of strings to vocabulary ids
    index.update([ (k, i) for i, k in enumerate(index_rev, start=vocab_start) ])
    index_size = vocab_start + len(index)  # largest vocabulary id + 1

    # add none/padding and out-of-vocabulary ids
    index[none_key] = none_ids
    index[oov_key] = oov_ids
    return index, index_size

import numpy as np
import random
from keras.engine.training import make_batches

def map_sequence(sequence, index, oov_key=""):
    """Map sequence of strings to vocabulary ids."""

    ids = []
    for s in sequence:
        try:
            ids.append(index[s])
        except KeyError:  # missing in vocabulary
            ids.append(index[oov_key])
    return ids


def pad_sequence(sequence, max_len, value=0, max_rand=None):
    """Post-pad sequence of ids as numpy array."""

    # crop sequence if needed
    sequence = sequence[:max_len]

    # convert to numpy array with masked and random post-padding
    if isinstance(value, int):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.int) * value])
    elif isinstance(value, float):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.float32) * value])
    elif value == 'rand' and isinstance(max_rand, int):
        x = np.hstack([sequence, np.random.randint(1, max_rand, size=max_len - len(sequence),)])
    else:
        raise ValueError("Padding value '{}' not understood".format(value))
    return x

def batch_generator(dataset, arg1_len, arg2_len, conn_len, punc_len, batch_size, random_per_sample=1):
    """Batch generator where each sample represents a different discourse relation."""

    batch_size = int(batch_size / (1.0 + random_per_sample))

    rel_ids = list(dataset['rel_ids'])  # copy list
    while True:
        # shuffle relations on each epoch
        random.shuffle(rel_ids)
        for batch_start, batch_end in make_batches(len(rel_ids), batch_size):

            # prepare batch data
            _rel_id = []
            data_in = {}
            data_out = {}
            for rel_id in rel_ids[batch_start:batch_end]:
                # choose sample
                doc_id = dataset['rel_parts'][rel_id]['DocID']
                words_len = len(dataset['words'][doc_id])

                _rel_id.append(rel_id)

                def tokens_np(token_ids, max_len):
                    words_slice = [ dataset['words'][doc_id][i]  for i in token_ids ]

                    ids = map_sequence(words_slice, indexes['words2id'])
                    x_np = pad_sequence(ids, max_len, value=0)
                    return x_np

                # arg1/2/conn/punc/rand inputs
                arg1_np = tokens_np(dataset['rel_parts'][rel_id]['Arg1'], arg1_len)
                try:
                    data_in['arg1_ids'].append(arg1_np)
                except KeyError:
                    data_in['arg1_ids'] = [arg1_np]

                arg2_np = tokens_np(dataset['rel_parts'][rel_id]['Arg2'], arg2_len)
                try:
                    data_in['arg2_ids'].append(arg2_np)
                except KeyError:
                    data_in['arg2_ids'] = [arg2_np]

                conn_np = tokens_np(dataset['rel_parts'][rel_id]['Connective'], conn_len)
                try:
                    data_in['conn_ids'].append(conn_np)
                except KeyError:
                    data_in['conn_ids'] = [conn_np]

                punc_np = tokens_np(dataset['rel_parts'][rel_id]['Punctuation'], punc_len)
                try:
                    data_in['punc_ids'].append(punc_np)
                except KeyError:
                    data_in['punc_ids'] = [punc_np]

                # relation senses output
                def rsenses_np(cat, oov_key=""):
                    try:
                        i = indexes['rel_senses2id'][cat]
                    except KeyError:  # missing in vocabulary
                        i = indexes['rel_senses2id'][oov_key]
                    x_np = np.zeros((indexes_size['rel_senses2id'],), dtype=np.float32)
                    x_np[i] = 1
                    return x_np

                rsenses = rsenses_np(dataset['rel_senses'][rel_id])
                try:
                    data_out['rsenses'].append(rsenses)
                except KeyError:
                    data_out['rsenses'] = [rsenses]

                # random noise for each sample
                for _ in range(random_per_sample):
                    j = np.random.randint(3)
                    if j == 0:
                        # random arg1
                        arg1_np = np.random.randint(1, 1 + np.max(arg1_np), size=arg1_np.shape)
                    elif j == 1:
                        # random arg2
                        arg2_np = np.random.randint(1, 1 + np.max(arg2_np), size=arg2_np.shape)
                    else:
                        # random arg1 and arg2
                        arg1_np = np.random.randint(1, 1 + np.max(arg1_np), size=arg1_np.shape)
                        arg2_np = np.random.randint(1, 1 + np.max(arg2_np), size=arg2_np.shape)
                    # random connective
                    if np.max(conn_np) > 0:
                        conn_np = np.random.randint(1, 1 + np.max(conn_np), size=conn_np.shape)
                    # random punctuation
                    if np.max(punc_np) > 0:
                        punc_np = np.random.randint(1, 1 + np.max(punc_np), size=punc_np.shape)
                    # mark as out-of-vocabulary in rsenses
                    rsenses = rsenses_np("")

                    _rel_id.append(rel_id)
                    data_in['arg1_ids'].append(arg1_np)
                    data_in['arg2_ids'].append(arg2_np)
                    data_in['conn_ids'].append(conn_np)
                    data_in['punc_ids'].append(punc_np)
                    data_out['rsenses'].append(rsenses)

            # convert to NumPy array
            for k, v in data_in.items():
                data_in[k] = np.asarray(v)
            for k, v in data_out.items():
                data_out[k] = np.asarray(v)

            # append meta data
            data_in['_rel_id'] = _rel_id

            # yield batch
            yield (data_in, data_out)

def load_word2vec(words2id, words2id_size, words_dim, words2vec_bin=None, words2vec_txt=None):
    if not words2vec_bin and not words2vec_txt:
        return None  # no pre-trained word embeddings

    import numpy as np
    from gensim.models import word2vec
    if words2vec_bin:
        model = word2vec.Word2Vec.load_word2vec_format(words2vec_bin, binary=True)
    else:
        model = word2vec.Word2Vec.load_word2vec_format(words2vec_txt)
    init_weights = np.zeros((words2id_size, words_dim), dtype=np.float32)
    for k, i in words2id.iteritems():
        if not isinstance(k, str):
            continue
        try:
            init_weights[i] = model[k][:words_dim]
        except KeyError:  # missing in word2vec
            pass
    return [init_weights]


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
words_dropout = c('words_dropout', 0.33)  #0-0.2?
focus_dropout_W = c('focus_dropout_W', 0.33)  #0?, >0.5?
focus_dropout_U = c('focus_dropout_U', 0.66)  #0?, irrelevant?
rnn_dropout_W = c('rnn_dropout_W', 0.33)  #0.6-0.8?, irrelevant?
rnn_dropout_U = c('rnn_dropout_U', 0.66)  #0-0.5?
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
train_snapshot = next(batch_generator(train, arg1_len, arg2_len, conn_len, punc_len, min(len(train['rel_ids']), snapshot_size), random_per_sample=0))
#    save_dict_of_np(train_snapshot_dir, train_snapshot)
#train_snapshot = load_dict_of_np(train_snapshot_dir)
#if not os.path.isdir(valid_snapshot_dir):
valid_snapshot = next(batch_generator(valid, arg1_len, arg2_len, conn_len, punc_len, min(len(valid['rel_ids']), snapshot_size), random_per_sample=0))
#    save_dict_of_np(valid_snapshot_dir, valid_snapshot)
#valid_snapshot = load_dict_of_np(valid_snapshot_dir)
train_iter = batch_generator(train, arg1_len, arg2_len, conn_len, punc_len, batch_size, random_per_sample=random_per_sample)

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
