#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Generic Python utils.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import cPickle as pickle
import numpy as np
import os


class Tee(object):
    """Redirect output streams to console and log files.

    > f = codecs.open("console.log", mode='a', encoding='utf8')
    > sys.stdout = Tee([sys.stdout], [f])
    """

    def __init__(self, direct=[], files=[]):
        self.direct = list(direct)
        self.files = list(files)
        self.buf = ""

    def write(self, obj):
        # direct output
        for f in self.direct:
            f.write(obj)

        # buffered line output to files
        self.buf += obj
        line = ""
        for line in self.buf.splitlines(True):
            if line.endswith("\n"):  # write only whole lines
                for f in self.files:
                    f.write(line)
                line = ""
        self.buf = line  # preserve last unflushed line

    def flush(self) :
        for f in self.direct + self.files:
            f.flush()


def debugger(type, value, tb):
    """Hook for handling exceptions using pdb debugger.

    > sys.excepthook = debugger
    """

    import sys, traceback, pdb
    sys.excepthook = None  # detach debugger
    traceback.print_exception(type, value, tb)
    pdb.pm()


def load_from_pkl(pkl):
    """Load experiment resource from file."""

    try:
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    except IOError:
        pass
    return None


def save_to_pkl(pkl, obj):
    """Save experiment resource to file."""

    with open(pkl, 'wb') as f:
        pickle.dump(obj, f)
    return obj


def load_dict_of_np(dir):
    """Load dictionary of NumPy arrays from read-only memory-mapped files."""

    data_tuple = []
    for pre in ["i", "t", "w"]:
        keys = load_from_pkl("{}/{}-keys.pkl".format(dir, pre))
        if keys is None:
            continue
        data = {}
        for k in keys:
            data[k] = np.load("{}/{}-{}.npy".format(dir, pre, k), mmap_mode='r')
        data_tuple.append(data)
    return data_tuple


def save_dict_of_np(dir, data_tuple):
    """Save dictionary of NumPy arrays to memory-mapped files."""

    os.makedirs(dir)
    for pre, data in zip(["i", "t", "w"], data_tuple):
        keys = data.keys()
        for k in keys:
            np.save("{}/{}-{}.npy".format(dir, pre, k), data[k])
        save_to_pkl("{}/{}-keys.pkl".format(dir, pre), keys)
    return data


# Tests

def test_save_load_pkl(tmpdir):
    pkl = str(tmpdir.join("test_save_load_pkl.pkl"))
    t_index = {None: 0, "": 1, "foo": 2, "bar": 3}
    t_index_size = len(t_index)
    t_obj = (t_index, t_index_size)

    obj = save_to_pkl(pkl, t_obj)
    assert obj == t_obj

    obj = load_from_pkl(pkl)
    assert obj == t_obj

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
