#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Execution objective with hyperopt.mongoexp.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import hashlib
import json
import os
import sys
import subprocess

DEVNULL = open(os.devnull, 'wb')


def objective(config):
    """Execute the learning script and use last line in JSON format as result."""

    config_str = json.dumps(config)
    exp_hash = hashlib.md5(config_str).hexdigest()
    exp_key = config.pop('_exp_key')
    cmd_fmt = config.pop('_cmd')

    config_str = json.dumps(config)
    cmd = cmd_fmt.format(exp_key=exp_key, exp_hash=exp_hash, config_str=config_str)
    print "experiment ({}-{})".format(exp_key, exp_hash)
    print "  config: {}".format(config_str)
    print "  cmd: {}".format(cmd)
    sys.stdout.flush()

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=DEVNULL)
    line = ""
    for line in iter(p.stdout.readline, ""):
        pass
    print "  last line:   {}".format(line)
    sys.stdout.flush()

    try:
        return json.loads(line.rstrip())
    except ValueError:
        return {
            'status': 'fail',
        }
