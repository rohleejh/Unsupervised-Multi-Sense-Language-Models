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



function make_stopword_TF(corpus_name, stage2dir)
	local d = torch.load(string.format('data/%s/data.t7',corpus_name))
	local vocab = torch.load(string.format('data/%s/vocab.t7', corpus_name))
	local freq = torch.Tensor(vocab.freq)

	sw = torch.zeros(#vocab.idx2word)
	word = torch.load(string.format('data/%s/stopword_nltk.t7',corpus_name))
	for i = 1,#word do
		local w = word[i]
		if vocab.word2idx[w] ~= nil then
			sw[ vocab.word2idx[w] ] = 1
		end
	end
	-- <eos> = +
	-- <unk> = |
	local except_word = {'N','+','|','.',',','!','"','-',';','?','*','(',')',':',"'",'[',']','/','<eos>','<unk>'}
	for i = 1, #except_word do
		local w = except_word[i]
		if vocab.word2idx[w] ~= nil then
			sw[ vocab.word2idx[w] ] = 1
		end
	end

	local thre = 50
	sw[freq:lt(thre)] = 1
	torch.save(string.format('data/%s/stopword_%s_TF.t7',corpus_name,corpus_name,thre),sw)
end

function make_new_vocab(num_cluster, corpus_name, stage2dir)
	local stopword_TF = torch.load(string.format('data/%s/stopword_%s_TF.t7',corpus_name,corpus_name))
	local vocab = torch.load(string.format('data/%s/vocab.t7',corpus_name))

	local freq = vocab.freq

	num_sent_per_word = {}
	num_sent_per_word[1] = {}
	num_sent_per_word[2] = {}
	num_sent_per_word[3] = {}
	---- new vocab setting
	word_start_index= {}
	idx2orgidx = {}
	idx2word = {}
	word2idx = {}
	-----------------------

	num = 0
	word = 0
	local cluster = num_cluster
	for i = 1,#vocab.idx2word do 
		local w = vocab.idx2word[i]
		if (stopword_TF[i] == 0) then 
			num = num+1
			num_sent_per_word[1][num] = i
			num_sent_per_word[2][num] = freq[i]
			num_sent_per_word[3][num] = cluster
			
			word_start_index[i] = word+1 -- word start index ******** important! 
			
			for j = 1, cluster do
				idx2orgidx[word+j] = i
				idx2word[word+j] = w
				word2idx[w] = word+j
			end
			word = word+cluster
		else
			word = word+1
			word_start_index[i] = word
			idx2orgidx[word] = i
			idx2word[word] = w
			word2idx[w] = word
		end
	end 
	num_sent_per_word = torch.Tensor(num_sent_per_word)
	word_start_index = torch.Tensor(word_start_index)
	torch.save(string.format('data/%s/vocab_c%d.t7',corpus_name,num_cluster),{freq=freq,idx2orgidx=idx2orgidx, idx2word=idx2word, word2idx=word2idx, org2stridx=word_start_index})
	torch.save(string.format('data/%s/num_sent_per_word.t7',corpus_name),num_sent_per_word)
	matio.save(string.format('%s/num_sent_per_word.mat',stage2dir),{num_sent_per_word = num_sent_per_word})
	torch.save(string.format('data/%s/word_start_index_c%d.t7',corpus_name,num_cluster),word_start_index)
end

function save_train_sentence(corpus_name, stage2dir)
	local d = torch.load(string.format('data/%s/data.t7',corpus_name))
	print(stage2dir, opt.num_layers,opt.word_vec_size, opt.rnn_size)
	local concat_file = string.format('%s/concat_sent_vec_lyr%d_hid%d%d', stage2dir, opt.num_layers,opt.word_vec_size, opt.rnn_size)
	print('Save context vector from training data to ', concat_file)
	
	local vocab = torch.load(string.format('data/%s/vocab.t7',corpus_name))
	local stopword_TF = torch.load(string.format('data/%s/stopword_%s_TF.t7',corpus_name,corpus_name))
	local freq = torch.Tensor(vocab.freq)
	
	--context(sentence) vector 
	rnn_size = opt.rnn_size
	sent = {}
	local num = 0
	for i = 1,#loader.idx2word do 
		if (stopword_TF[i] == 0) then
			sent[i] = torch.Tensor(freq[i],opt.rnn_size):fill(0)
			num = num+1
		else
			sent[i] = {}
		end
	end 
	print('multi-sense words: ',num)
		
	-------- Save context vector from LSTM hidden
	local rnn_state = {[0] = init_state}
	local tot_num = 0
	x, y = loader:next_batch(1)
	x = torch.repeatTensor(d[1],opt.batch_size,1)
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	end

	local sent_idx_tmp = {}
	local len = {}
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2)-1 do
	    lst = protos.rnn:forward(get_input(x, t, rnn_state[0]))
	    if (stopword_TF[x[1][t+1]] == 0) then 
	    	if len[x[1][t+1]] == nil then
	    		len[x[1][t+1]] = 1
	    	else
	    		len[x[1][t+1]] = len[x[1][t+1]] + 1
	    	end
	    	sent[x[1][t+1]][len[x[1][t+1]]] = lst[2*opt.num_layers][1]:clone():float() -- need to be clone (hidden value: lst[2][1], cell: lst[1][1])
	    	tot_num = tot_num + 1
	    end
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    if x[1][t+1] == loader.word2idx[opt.EOS] then
	    	rnn_state = {[0] = init_state} -- reset hidden, cell state
	    end
	    if t%100000 == 0 then
			print(string.format('save context vector ... %d, tot # data = %d ',t, tot_num))
	    end
	end
	print(string.format('save context vector ... %d, tot # data = %d ',x:size(2)-1, tot_num))
	
	----- concatenate sentence vector
	num = 0
	for i = 1,#loader.idx2word do 
		if (stopword_TF[i] == 0) then
			num = num+1
			if num == 1 then
				x = sent[i]
			else
				x = torch.cat(x,sent[i], 1)
			end
			if num%100 == 0 then
				print(string.format('concat context vector ... %d, word idx = %d ',num, i))
			end
		end
	end 
	
	print(string.format('concat sent vector size: %d  ',x:size(1)))
	local matio = require 'matio'
	matio.save(concat_file .. '.mat',{sent_all = x})
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

	if not path.exists(stage2dir) then lfs.mkdir(stage2dir) end

	make_stopword_TF(corpus_name, stage2dir)
	make_new_vocab(num_cluster, corpus_name, stage2dir)
	save_train_sentence(corpus_name, stage2dir) 
end

load_network()
