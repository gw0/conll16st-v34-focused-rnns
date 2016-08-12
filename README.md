conll16st-v34-focused-rnns
==========================

System implementation of the paper Discourse Sense Classification from Scratch using Focused RNNs (presented at *CoNLL 2016* conference). Implemented in Python 2 using Numpy, Keras with Theano.

**Note:** Same implementation was used on English and Chinese datasets. It achieved new state-of-the-art on Chinese blind dataset.

- <http://www.cs.brandeis.edu/~clp/conll16st/>
- <http://github.com/attapol/conll16st/>
- <http://github.com/gw0/conll16st-v34-focused-rnns/>

Check out the **conference paper** and **presentation** at:

- <http://gw.tnode.com/deep-learning/conll2016-discourse-sense-classification-from-scratch-using-focused-rnns/>


Abstract
--------

The subtask of *CoNLL 2016 Shared Task* focuses on sense classification of multilingual shallow discourse relations. Existing systems rely heavily on external resources, hand-engineered features, patterns, and complex pipelines fine-tuned for the English language. In this paper we describe a different approach and system inspired by end-to-end training of deep neural networks. Its input consists of only sequences of tokens, which are processed by our novel focused RNNs layer, and followed by a dense neural network for classification. Neural networks implicitly learn latent features useful for discourse relation sense classification, make the approach almost language-agnostic and independent of prior linguistic knowledge. In the closed-track sense classification task our system achieved overall *0.5246* F1-measure on English blind dataset and achieved the new state-of-the-art of *0.7292* F1-measure on Chinese blind dataset.

<figure><img src="http://gw.tnode.com/deep-learning/img/conll16st-v34-model.png" width="600" height="484" alt="Our CoNLL 2016 Shared Task individual discourse sense classifier/model." /></figure>


Usage
-----

Script for applying both trained models for English and Chinese that were used on TIRA system (check its source code):

```bash
# tira_run_{en|zh}.sh <dataset_dir> <output_dir>
$ ./v34/tira_run_en.sh ./data/conll16st-en-03-29-16-trial ./output
$ ./v34/tira_run_zh.sh ./data/conll16st-zh-01-08-2016-trial ./output
```

For training each individual model use:

```bash
# train.py <experiment_dir> <train_dir> <valid_dir> [--clean] [--config CONFIG]
$ ./v34/train.py ./models-v34-a ./data/conll16st-en-03-29-16-train ./data/conll16st-en-03-29-16-dev --config='{"filter_fn_name":"conn_eq_0"}'
```

Afterwards apply the trained model to an unseen dataset with:

```bash
# classifier.py <lang> <model_dir> <dataset_dir> <output_dir> [--config CONFIG]
$ ./v34/classifier.py en ./models-v34-a ./data/conll16st-en-03-29-16-test ./output --config='{"filter_fn_name":"conn_eq_0"}'
```

For evaluation use the [official CoNLL 2016 Shared Task scorer](http://github.com/attapol/conll16st):

```bash
$ ./conll16st_evaluation/tira_sup_eval.py ./data/conll16st-en-03-29-16-test ./output ./output
```


License
=======

Copyright &copy; 2016 *gw0* [<http://gw.tnode.com/>] &lt;<gw.2016@tnode.com>&gt;

This code is licensed under the [GNU Affero General Public License 3.0+](LICENSE_AGPL-3.0.txt) (*AGPL-3.0+*). Note that it is mandatory to make all modifications and complete source code publicly available to any user.
