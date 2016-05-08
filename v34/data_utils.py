#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Dataset preparing utils.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import numpy as np
import random
from keras.engine.training import make_batches


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

def batch_generator(dataset, indexes, indexes_size, arg1_len, arg2_len, conn_len, punc_len, batch_size, random_per_sample=1):
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

                if dataset['rel_senses']:
                    rsenses = rsenses_np(dataset['rel_senses'][rel_id])
                else:
                    rsenses = rsenses_np("")
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


def decode_relation(y_np, cats2id, cats2id_size):
    """Decode categories from one-hot vector (sample, cats2id)."""

    # normalize by rows to [0,1] interval
    y_sum = np.sum(y_np)
    totals = y_np / y_sum
    totals[y_sum == 0.] = y_np[y_sum == 0.]  # prevent NaN

    # return most probable category
    cat = None
    max_total = -1.
    for t, j in cats2id.items():
        if totals[j] > max_total:
            max_total = totals[j]
            cat = t
    return cat, totals


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
