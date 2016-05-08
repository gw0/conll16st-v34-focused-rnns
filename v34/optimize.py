#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Distributed hyper-parameter optimization with hyperopt.mongoexp.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import argparse
import logging
import os
from hyperopt import hp, fmin, tpe, space_eval
from hyperopt.mongoexp import MongoTrials, main_worker_helper, spec_from_misc

import optimize_exec


def list_best(mongo, exp_key=None, space=None):
    mongo_trials = MongoTrials(mongo, exp_key=exp_key)

    jobs = mongo_trials.trials
    jobs_ok = [ (d['result']['loss'], d)  for d in jobs if d['state'] == 2 and d['result']['status'] == 'ok']
    jobs_ok.sort()

    for loss, job in reversed(jobs_ok):
        print(loss, job['owner'], job['result'])
        spec = spec_from_misc(job['misc'])
        print("spec:  {}".format(spec))
        if space is not None:
            print("space: {}".format(space_eval(space, spec)))
    print("total: {}/{}".format(len(jobs_ok), len(jobs)))
    return mongo_trials.argmin


# logging
log = logging.getLogger("hyperopt")
if log:
    import sys
    log.addHandler(logging.StreamHandler(stream=sys.stdout))
    log.setLevel(logging.DEBUG)

# parse arguments
argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('action',
    choices=['worker', 'optimizer', 'list_best'],
    help="action to perform")
argp.add_argument('--mongo',
    default="mongo://conll16st-mongo:27017/conll16st/jobs",
    help="mongo connection string")
argp.add_argument('--exp-key',
    default=None,
    help="identifier for optimization experiments")
argp.add_argument('--evals', type=int,
    default=10,
    help="maximal number of evaluations (for optimizer)")
argp.add_argument('--cmd',
    default="/srv/v34/train.py /srv/ex/{exp_key}-{exp_hash} /srv/data/conll16st-zh-train /srv/data/conll16st-zh-dev --clean --config='{config_str}'",
    help="command for each experiment (for optimizer)")
argp.add_argument('--worker-helper',
    default="/usr/local/bin/hyperopt-mongo-worker",
    help="command for worker helper (for worker)")
args = argp.parse_args()

# define configuration search space
space = {
    '_cmd': args.cmd,
    '_exp_key': args.exp_key,
    'filter_fn_name': "conn_gt_0",  #!!!: "conn_eq_0", "conn_gt_0"
    'epochs': 200,
    'epochs_len': -1,
    'epochs_patience': 10,
    #'batch_size': 64,
    #'snapshot_size': 2048,
    'random_per_sample': hp.quniform('random_per_sample', 8, 64, 8.),
    'words_dim': hp.quniform('words_dim', 10, 100, 10.),
    'focus_dim': hp.quniform('focus_dim', 2, 8, 1.),
    'rnn_dim': hp.quniform('rnn_dim', 10, 100, 10.),
    'final_dim': hp.quniform('final_dim', 10, 100, 10.),
    'arg1_len': 500,  #= 100 (en), 500 (zh)
    'arg2_len': 500,  #= 100 (en), 500 (zh)
    #'conn_len': 10,
    #'punc_len': 2,
    'words_dropout': hp.quniform('words_dropout', 0.0, 1.0, 0.25),
    'focus_dropout_W': hp.quniform('focus_dropout_W', 0.0, 1.0, 0.25),
    'focus_dropout_U': hp.quniform('focus_dropout_U', 0.0, 1.0, 0.25),
    'rnn_dropout_W': hp.quniform('rnn_dropout_W', 0.0, 1.0, 0.25),
    'rnn_dropout_U': hp.quniform('rnn_dropout_U', 0.0, 1.0, 0.25),
    'final_dropout': hp.quniform('final_dropout', 0.0, 1.0, 0.25),
    'words2vec_bin': None,
    'words2vec_txt': None,
    #'rsenses_imp_loss': "categorical_crossentropy",
}

if args.action == 'worker':
    # run distributed worker
    class Options(object):
        mongo = args.mongo
        exp_key = args.exp_key
        last_job_timeout = None
        max_consecutive_failures = 2
        max_jobs = args.evals  #= inf
        poll_interval = 300  #= 5 min
        reserve_timeout = 3600  #= 1 hour
        workdir = None
    sys.argv[0] = args.worker_helper
    sys.exit(main_worker_helper(Options(), None))

elif args.action == 'optimizer':
    # run distributed optimizer
    trials = MongoTrials(args.mongo, exp_key=args.exp_key)
    best = fmin(optimize_exec.objective, space, trials=trials, algo=tpe.suggest, max_evals=args.evals)

    # summary
    print
    print "evals: {}".format(args.evals)
    print "best:  {}".format(best)
    print "space: {}".format(space_eval(space, best))

elif args.action == 'list_best':
    # list distributed evaluation results
    best = list_best(args.mongo, exp_key=args.exp_key, space=space)

    # summary
    print
    print "evals: {}".format(args.evals)
    print "best:  {}".format(best)
    print "space: {}".format(space_eval(space, best))

else:
    raise Exception("Invalid action '{}'".format(args.action))
