--[[
Much of the code is borrowed from the following implementations
https://github.com/yoonkim/lstm-char-cnn
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm

Trains a single-sense or multi-sense lstm language model
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
local matio = require 'matio'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a lstm language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
cmd:option('-corpus_name','ptb','Corpus name.')
cmd:option('-stage2dir','stage2_file_ptb','data directory for Stage 2.')
-- model params
cmd:option('-num_cluster', 1, 'number of clusters per word in stage 1 or 3')
cmd:option('-num_cluster_in_stage2', 3, 'number of clusters per word')
cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
-- optimization
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-optim_method','sgd','optimization method : (sgd, rmsprop, adagrad, adam )')
cmd:option('-momentum', 0.9,'momentum for sgd')
cmd:option('-wdecay', 0,'weight dcay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-seq_length',35,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 5, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-EOS', '<eos>', 'end of sentence symbol.')
cmd:option('-time', 0, 'print batch times')
cmd:option('-model_dir', '', 'filename to load model')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.EOS = opt.EOS
opt.tokens.UNK = '<unk>' -- unk word token

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length)
print('Word vocab size: ' .. #loader.idx2word)

LSTM = require 'model.LSTM'

protos = {}
print('creating an LSTM with ' .. opt.num_layers .. ' layers')
local output_size = #loader.idx2word
protos.rnn = LSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout,#loader.idx2word, opt.word_vec_size, opt.batch_norm)
protos.criterion = nn.ClassNLLCriterion()

-- Optimization
function set_optim(optim_method, learning_rate, wdecay, momentum)
  if optim_state == nil then optim_state = {} end
	  optim_state.learningRate = learning_rate
	  optim_state.weightDecay = wdecay
  if opt.optim_method == 'adam' then
    optim_state.beta1 = 0.9
    optim_state.beta2 = 0.999
  elseif opt.optim_method == 'sgd' then
    optim_state.momentum = momentum
  end  
end
set_optim(opt.optim_method, opt.learning_rate, opt.wdecay, opt.momentum)


if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
    
end

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
	local tn = torch.typename(layer)
	if layer.name ~= nil then
		if layer.name == 'word_vecs' then
			word_vecs = layer
		end
	end
end 

function get_input(x, t, prev_states)
    local u = {}
	table.insert(u, x[{{},t}])
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

function data_index2cluster_index(num_cluster, corpus_name, stage2dir)
	-- num_cluster is 3
	local data_file = string.format('data/%s/data_c%d.t7',corpus_name,num_cluster) --new file

	local c = matio.load(string.format('%s/centroid_lyr%d_h%d%d_c%d.mat',stage2dir,opt.num_layers,opt.word_vec_size, opt.rnn_size,num_cluster))
	local idx = matio.load(string.format('%s/index_lyr%d_h%d%d_c%d.mat',stage2dir,opt.num_layers,opt.word_vec_size, opt.rnn_size,num_cluster))
	
	local s_idx = torch.load(string.format('data/%s/word_start_index_c%d.t7',corpus_name,num_cluster))
	local fr_w = torch.load(string.format('data/%s/num_sent_per_word.t7',corpus_name))
	local v = torch.load(string.format('data/%s/vocab.t7',corpus_name))

	local map_org2tmp = {}
	for i = 1, fr_w:size(2) do
		map_org2tmp[fr_w[1][i]] = i
	end
	local x_new = {}

	local stopword_TF = torch.load(string.format('data/%s/stopword_%s_TF.t7',corpus_name,corpus_name))
	
	----- Load train/valid/test data
	local d = torch.load(string.format('data/%s/data.t7',corpus_name)) --Original data
	local x = torch.repeatTensor(d[1],opt.batch_size,1)
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	    c.centroid = c.centroid:float():cuda()
	end
	
	----- Train
	local tot_num = 0
	x_new[1] = {}
	local len = {}
	x_new[1][1] = s_idx[x[1][1]]
	for t = 1, x:size(2)-1 do
	    if (stopword_TF[x[1][t+1]] == 0) then
	    	if len[x[1][t+1]] == nil then
	    		len[x[1][t+1]] = 1
	    	else
	    		len[x[1][t+1]] = len[x[1][t+1]] + 1
	    	end
	    	
	    	local j = map_org2tmp[x[1][t+1]] -- from word Hot(565) to {number}
	    	local c_idx_tmp = fr_w[{2,{1,math.max(j-1,1)}}]:sum()
			if j == 1 then 
				c_idx_tmp = 0
			end
	    	x_new[1][t+1] = s_idx[x[1][t+1]] + idx.index[c_idx_tmp + len[x[1][t+1]]][1] - 1 
	    	tot_num = tot_num + 1
	    else
	    	x_new[1][t+1] = s_idx[x[1][t+1]]
	    end
	    if t%100000 == 0 then
			 print(string.format('save Train data ... %d, tot # data = %d ',t, tot_num))
	    end
	end
	print(string.format('save Train data ... %d, tot # data = %d ',x:size(2)-1, tot_num))
	x_new[1] = torch.Tensor(x_new[1]):long()
	torch.save(data_file,x_new)	
	
	----- Valid
	local rnn_state = {[0] = init_state}
	x=nil
	x_new[2] = {}
	local tot_num = 0
	local len = {}
	local eos = v.word2idx[opt.EOS]
	local eos_t = eos*torch.ones(opt.batch_size,1):long()
	local x = torch.repeatTensor(d[2],opt.batch_size,1)
	x = torch.cat(eos_t,x,2) --cat column
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	end
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2)-1 do
	    lst = protos.rnn:forward(get_input(x, t, rnn_state[0]))
	    if (stopword_TF[x[1][t+1]] == 0) then
	    	if len[x[1][t+1]] == nil then
	    		len[x[1][t+1]] = 1
	    	else
	    		len[x[1][t+1]] = len[x[1][t+1]] + 1
	    	end
	    	local j = map_org2tmp[x[1][t+1]] -- from word Hot(565) to {number}	    	
	    	sent = lst[2*opt.num_layers][1]:clone()-- need to be clone (hidden value)
	    	-- calculate euclidean distance 
	    	local dist = torch.Tensor(num_cluster)
	    	
			for k = 1, num_cluster do
				dist[k]=torch.dist(c.centroid[num_cluster*(j-1)+k], sent)
				y,i = torch.min(dist,1)
	    	end
	    	----------------------------
	    	x_new[2][t] = s_idx[x[1][t+1]] + i[1] - 1
	    	tot_num = tot_num + 1
	    else
	    	x_new[2][t] = s_idx[x[1][t+1]]
	    end
	    
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    if x[1][t+1] == loader.word2idx[opt.EOS] then
	    	rnn_state = {[0] = init_state} -- reset hidden, cell state
	    end
	    if t%100000 == 0 then
			 print(string.format('save Valid data ... %d, tot # data = %d ',t, tot_num))
	    end
	end
	print(string.format('save Valid data ... %d, tot # data = %d ',x:size(2)-1, tot_num))
	x_new[2] = torch.Tensor(x_new[2]):long()
	torch.save(data_file,x_new)
	
	----- Test
	rnn_state = {[0] = init_state}
	x = nil
	x_new[3] = {}
	local tot_num = 0
	local len = {}
	local x = torch.repeatTensor(d[3],opt.batch_size,1)
	x = torch.cat(eos_t,x,2) --cat column
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	end
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2)-1 do
	    lst = protos.rnn:forward(get_input(x, t, rnn_state[0]))
	    if (stopword_TF[x[1][t+1]] == 0) then
	    	if len[x[1][t+1]] == nil then
	    		len[x[1][t+1]] = 1
	    	else
	    		len[x[1][t+1]] = len[x[1][t+1]] + 1
	    	end
	    	local j = map_org2tmp[x[1][t+1]] -- from word Hot(565) to {number}
	    	sent = lst[2*opt.num_layers][1]:clone() -- need to be clone (hidden value)
	    	
	    	-- calculate euclidean distance 
	    	local dist = torch.Tensor(num_cluster)
			for k = 1, num_cluster do
				dist[k]=torch.dist(c.centroid[num_cluster*(j-1)+k], sent)
				y,i = torch.min(dist,1)
	    	end
	    	
	    	----------------------------
	    	x_new[3][t] = s_idx[x[1][t+1]] + i[1] - 1
	    	tot_num = tot_num + 1
	    else
	    	x_new[3][t] = s_idx[x[1][t+1]]
	    end
	    
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    if x[1][t+1] == loader.word2idx[opt.EOS] then
	    	rnn_state = {[0] = init_state} -- reset hidden, cell state
	    end
	    if t%100000 == 0 then
			print(string.format('save Test data ... %d, tot # data = %d ',t, tot_num))
	    end
	end
	print(string.format('save Test data ... %d, tot # data = %d ',x:size(2)-1, tot_num))
	x_new[3] = torch.Tensor(x_new[3]):long()
	torch.save(data_file,x_new)
	
end

function load_network()
	local num_cluster = opt.num_cluster_in_stage2
	local stage2dir = opt.stage2dir
	local corpus_name = opt.corpus_name

	print('load model from ' .. opt.model_dir)
	checkpoint = torch.load(opt.model_dir)

	protos = checkpoint.protos  
	opt = checkpoint.opt
	opt.EOS = '<eos>'
	epoch = checkpoint.epoch + 1 
	train_losses = checkpoint.train_losses
	val_loss = checkpoint.val_loss
	val_losses = checkpoint.val_losses
	i = checkpoint.i
	lr = checkpoint.lr
	set_optim(opt.optim_method, lr, opt.wdecay, opt.momentum)

	init_state = {}
	for L=1,opt.num_layers do
		local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
		if opt.gpuid >=0 then h_init = h_init:cuda() end
		table.insert(init_state, h_init:clone())
		table.insert(init_state, h_init:clone())
	end
    for k,v in pairs(protos) do v:cuda() end
	
	params, grad_params = model_utils.combine_all_parameters(protos.rnn)
	print('number of parameters in the model: ' .. params:nElement())
	protos.rnn:apply(get_layer)

	clones = {}
	for name,proto in pairs(protos) do
		 print('cloning ' .. name)
		 clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
	end
	
	print("epoch: " .. epoch .. " train_losses: " .. train_losses[#train_losses] )  

	checkpoint=nil  
	collectgarbage()

	data_index2cluster_index(num_cluster, corpus_name, stage2dir)
end

load_network()
