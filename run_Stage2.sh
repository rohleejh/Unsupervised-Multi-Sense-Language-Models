#!/bin/bash

#add -gpuid 0 to use GPU 

gpu=0
num_clusters=3
###------------1) Extract context vector from SSLM--LSTM model.
dir='baseline_model'
data_dir='data/ptb'
#-------------------------------------------------------
wd=650
rsize=650
lyr=2
lr=1
mom=0
schedule=0.5
wdcay=0
max_grad_norm=5
print_every=500
batch_size=20
dropout=0.5
num_cluster_in_stage2=${num_clusters}

savefile="${lyr}lyr-h${wd}${rsize}-lr${lr}"
modeldir="${dir}/lm_${savefile}.t7"

th save_context_vectors.lua -data_dir ${data_dir} -num_cluster_in_stage2 ${num_cluster_in_stage2} -checkpoint_dir ${dir} -savefile ${savefile} -num_layers ${lyr} -word_vec_size ${wd} -rnn_size ${rsize} -gpuid ${gpu} -momentum ${mom} -model_dir ${modeldir}

###------------2) K-means clustering on context vectors with K=3
echo "Run K-means clustering ..."
num_cluster=${num_clusters}
exc_code="Kmeans_total(${num_cluster});exit;"
matlab -nodisplay -nosplash -nodesktop -nojvm -r ${exc_code} | tail -n +10

###------------3) Re-index single-sense tokens to multi-sense tokens
th generate_ms_tokens.lua -data_dir ${data_dir} -num_cluster_in_stage2 ${num_cluster_in_stage2} -checkpoint_dir ${dir} -savefile ${savefile} -num_layers ${lyr} -word_vec_size ${wd} -rnn_size ${rsize} -gpuid ${gpu} -momentum ${mom} -model_dir ${modeldir}
