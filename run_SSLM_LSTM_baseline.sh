#!/bin/bash

#run make_text_to_tensor.lua to get data.t7
#add -gpuid 0 to use GPU 

# LSTM with hidden size 650
# Baseline (Train)
gpu=0
dir='baseline_model'
data_dir='data/ptb'
######################## 
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

savefile="${lyr}lyr-h${wd}${rsize}-lr${lr}"
th main.lua -data_dir ${data_dir} -checkpoint_dir ${dir} -savefile ${savefile} -batch_size ${batch_size} -dropout ${dropout} -num_layers ${lyr} -word_vec_size ${wd} -rnn_size ${rsize} -EOS '<eos>' -gpuid ${gpu} -momentum ${mom} -learning_rate ${lr} -learning_rate_decay ${schedule} -wdecay ${wdcay} -max_grad_norm ${max_grad_norm} -print_every ${print_every}
