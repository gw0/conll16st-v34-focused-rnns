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

FROM gw000/keras:1.0.1-py2
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>

# requirements (for project)
RUN pip install gensim pattern
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
