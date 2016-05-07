#!/bin/bash
# Requirements installation for conll16st-multi-classifier-v30
#
# Author: gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>
# License: All rights reserved

NAME="(`basename $(realpath ${0%/*})`)"
SRC="venv/src"
PYTHON_EXE="python2"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'

# install essentials
set -e
sudo() { [ -x "/usr/bin/sudo" ] && /usr/bin/sudo "$@" || "$@"; }
sudo apt-get install -y python-pip python-virtualenv build-essential g++ git

# create virtualenv
cd "${0%/*}"
virtualenv --system-site-packages --prompt="$NAME" --python="$PYTHON_EXE"  venv || exit 1
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"

# requirements for theano
sudo apt-get install -y gfortran python-dev libopenblas-dev liblapack-dev
#sudo apt-get install -y python-numpy python-scipy
#[ ! -d $SITE_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
#[ ! -d $SITE_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# requirements for keras
sudo apt-get install -y python-h5py python-yaml graphviz
#[ ! -d $SITE_PACKAGES/h5py ] && cp -a $DIST_PACKAGES/h5py* $SITE_PACKAGES

# requirements for matplotlib
sudo apt-get install -y pkg-config libpng-dev libfreetype6-dev

# general requirements (in Dockerfile)
pip install pydot-ng pyparsing matplotlib
pip install git+https://github.com/Theano/Theano.git@rel-0.8.1
pip install git+https://github.com/fchollet/keras.git@1.0.1

# requirements (for project)
pip install gensim pattern
git clone https://github.com/gw0/conll16st_data.git "$SRC/conll16st_data" && ln -s "$SRC/conll16st_data" ./conll16st_data
git clone https://github.com/attapol/conll16st.git "$SRC/conll16st" && ln -s "$SRC/conll16st" ./conll16st_evaluation

#XXX: patch Keras
cp patch_topology.py ./venv/lib/python2.7/site-packages/keras/engine/topology.py
cp patch_training.py ./venv/lib/python2.7/site-packages/keras/engine/training.py
cp patch_visualize_util.py ./venv/lib/python2.7/site-packages/keras/utils/visualize_util.py

# for hyper-parameter optimization
pip install git+http://github.com/vilcenzo/hyperopt.git
pip install networkx pymongo

# for development
pip install pytest

echo
echo "Use: . venv/bin/activate"
echo
