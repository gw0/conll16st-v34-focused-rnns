# Discourse relation sense classification (CoNLL16st).
#
# Example:
#   PRE=conll16st-v3403 PREDIR=/srv/storage/conll16st MEM=9000M DOCKER_ARGS="-m $MEM --memory-swap $MEM -v $PREDIR/data:/srv/data -v $PREDIR/ex:/srv/ex"
#   DATAT=en-train DATAV=en-dev DATAX=en-trial CONFIG='"_":0'
#     CONFIG='"words2vec_bin":"./data/word2vec-en/GoogleNews-vectors-negative300.bin.gz"'
#   DATAT=zh-train DATAV=zh-dev DATAX=zh-trial CONFIG='"arg1_len":500, "arg2_len":500'
#     CONFIG='"arg1_len":200, "arg2_len":200, "words2vec_txt":"./data/word2vec-zh/zh-Gigaword-300.txt"'
#   docker build -t $PRE /srv/repos/conll16st-v30
#
#   NAME=$PRE-w20-$DATAT
#   docker run -d $DOCKER_ARGS --name $NAME $PRE ./v34/train.py ex/$NAME data/conll16st-$DATAT data/conll16st-$DATAV --clean --config="{\"words_dim\":20, $CONFIG}" && echo -ne "\ek${NAME:10}\e\\" && sleep 5 && less +F $PREDIR/ex/$NAME/console.log
#     docker logs -f $NAME
#     docker rm -f $NAME
#     THEANO_FLAGS='mode=FAST_COMPILE'
#     THEANO_FLAGS='device=gpu2,floatX=float32,nvcc.fastmath=True,lib.cnmem=1'
#
#   NAME=$PRE-optimizer
#   docker $(weave config) run -d -m 100M --name $NAME $PRE ./v34/optimize.py optimizer --mongo=mongo://conll16st-mongo:27017/conll16st/jobs --exp-key=$PRE --evals=40 && echo -ne "\ek${NAME:10}\e\\" && docker logs -f $NAME
#
#   for ip in $(weave dns-lookup docker-vm); do echo -e "\n\n=== docker-vm : $ip ==="; ssh -o StrictHostKeyChecking=no $ip "docker ps -af name=$PRE; docker images $PRE; ps aux | grep '[d]ocker build'; cd /srv/storage/conll16st/data; echo conll16st-*"; done
#   for ip in $(weave dns-lookup docker-vm); do echo -e "\n\n=== docker-vm : $ip ==="; if ssh -o StrictHostKeyChecking=no $ip "test \! -d /srv/storage/conll16st/data"; then echo "copying data..."; ssh -o StrictHostKeyChecking=no $ip "mkdir -p /srv/storage/conll16st /srv/storage/conll16st/ex /srv/storage/conll16st/data; chmod 777 /srv/storage/conll16st/ex"; scp -qr -o StrictHostKeyChecking=no /srv/storage/conll16st/data/* $ip:/srv/storage/conll16st/data; fi; done
#   for ip in $(weave dns-lookup docker-vm); do echo -e "\n\n=== docker-vm : $ip ==="; ssh -o StrictHostKeyChecking=no $ip "mkdir -p /srv/repos/conll16st-v30"; scp -qr -o StrictHostKeyChecking=no /srv/repos/conll16st-v30/* $ip:/srv/repos/conll16st-v30; ssh -o StrictHostKeyChecking=no $ip "nohup docker build -t $PRE /srv/repos/conll16st-v30 > /dev/null 2>&1 &"; done
#   for ip in $(weave dns-lookup docker-vm); do echo -e "\n\n=== docker-vm : $ip ==="; ssh -o StrictHostKeyChecking=no $ip "docker \$(weave config) run -d $DOCKER_ARGS --name $PRE-W$ip $PRE ./v34/optimize.py worker --mongo=mongo://conll16st-mongo:27017/conll16st --exp-key=$PRE"; done
#   for ip in $(weave dns-lookup docker-vm); do echo -e "\n\n=== docker-vm : $ip ==="; ssh -o StrictHostKeyChecking=no $ip "docker rm -f \$(docker ps -aqf name=$PRE-W*); rm -rf /srv/storage/conll16st/ex/*"; done
#


FROM gw000/keras:1.0.1-py2
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>

# requirements (for project)
RUN pip install gensim pattern
RUN pip install git+http://github.com/vilcenzo/hyperopt.git
RUN pip install networkx pymongo
RUN git clone https://github.com/gw0/conll16st_data.git ./conll16st_data
RUN git clone https://github.com/attapol/conll16st.git ./conll16st_evaluation

# setup parser
ADD v34/ ./v34/
RUN useradd -r -d /srv parser \
 && mkdir -p /srv/ex \
 && chown -R parser:parser /srv \
 && ln -s /srv/v34/optimize_exec.py /usr/local/bin/

#XXX: patch Keras
ADD patch_topology.py /usr/local/lib/python2.7/dist-packages/keras/engine/topology.py
ADD patch_training.py /usr/local/lib/python2.7/dist-packages/keras/engine/training.py
ADD patch_visualize_util.py /usr/local/lib/python2.7/dist-packages/keras/utils/visualize_util.py

# expose interfaces
VOLUME /srv/data
VOLUME /srv/ex

USER parser
#ENTRYPOINT ["/usr/bin/python"]
#CMD ["/srv/v34/train.py", "ex/v3400", "data/conll16st-en-trial", "data/conll16st-en-trial", "--clean"]
