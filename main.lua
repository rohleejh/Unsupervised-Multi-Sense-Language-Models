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
cmd:option('-exp_ver','','experiment version')
-- model params
cmd:option('-num_cluster', 1, 'number of clusters per word')
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
cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 5, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-EOS', '<eos>', 'end of sentence symbol.')
cmd:option('-time', 0, 'print batch times')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
print(opt)
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
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.exp_ver)
print('Word vocab size: ' .. #loader.idx2word)

-- load model object
LSTM = require 'model.LSTM'

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}

print('creating an LSTM with ' .. opt.num_layers .. ' layers')

local output_size = #loader.idx2word
protos.rnn = LSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout,#loader.idx2word, opt.word_vec_size, opt.batch_norm)
-- training criterion (negative log likelihood)
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

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
print('number of parameters in the model: ' .. params:nElement())

-- initialization
params:uniform(-opt.param_init, opt.param_init) -- small numbers uniform

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
	local tn = torch.typename(layer)
	if layer.name ~= nil then
		if layer.name == 'word_vecs' then
			word_vecs = layer
		end
	end
end 
protos.rnn:apply(get_layer)
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for layer_idx = 1, opt.num_layers do
	for _, node in ipairs(protos.rnn.forwardnodes) do
		if node.data.annotations.name == "i2h_"..layer_idx then
			print('setting forget gate biases to 1 in LSTM layer '..layer_idx)
			-- the gates are, in order, i,o,f,g, so f is the 2nd block of weights
			node.data.module.bias[{{2*opt.rnn_size + 1, 3*opt.rnn_size}}]:fill(1.0)
		end
	end
end

-- make a bunch of clones after flattening, as that reallocates memory
-- not really sure how this part works
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

function get_input(x, t, prev_states)
    local u = {}
    table.insert(u, x[{{},t}])
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end


-- evaluate the loss over an entire split
function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]

    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local token_perp
    local loss = 0
	local uni_loss = 0
    local rnn_state = {[0] = init_state}    
    if split_idx<=2 then -- batch eval        
        for i = 1,n do -- iterate over batches in the split
            -- fetch a batch
            local x, y = loader:next_batch(split_idx)
            if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
                x = x:float():cuda()
                y = y:float():cuda()
            end
            -- forward pass
            for t=1,opt.seq_length do
                clones.rnn[t]:evaluate() -- for dropout proper functioning
                local lst = clones.rnn[t]:forward(get_input(x, t, rnn_state[t-1]))
                rnn_state[t] = {}
                for i=1,#init_state do 
                    table.insert(rnn_state[t], lst[i])
                end
                prediction = lst[#lst]
                loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
            end
            -- carry over lstm state
            rnn_state[0] = rnn_state[#rnn_state]
        end
        loss = loss / opt.seq_length / n
    else -- full eval on test set
        token_perp = torch.zeros(#loader.idx2word, 2) 
        local x, y = loader:next_batch(split_idx)
			if opt.gpuid >= 0 then -- ship the input arrays to GPU
                -- have to convert to float because integers can't be cuda()'d
                x = x:float():cuda()
                y = y:float():cuda()
			end
			protos.rnn:evaluate() -- just need one clone
			for t = 1, x:size(2) do
                local lst = protos.rnn:forward(get_input(x, t, rnn_state[0]))
                rnn_state[0] = {}
                for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
                prediction = lst[#lst] 
                tok_perp = protos.criterion:forward(prediction, y[{{},t}])
                uni_perp = torch.log(loader.unigram[y[1][t]])
                loss = loss + tok_perp
                uni_loss = uni_loss + tok_perp + uni_perp
                token_perp[y[1][t]][1] = token_perp[y[1][t]][1] + 1 --count
                token_perp[y[1][t]][2] = token_perp[y[1][t]][2] + tok_perp
			end
			loss = loss / x:size(2)
			uni_loss = uni_loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    local uniperp = torch.exp(uni_loss)
    return perp, token_perp, uniperp
end
function isnan(x) return x~=x end
-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)  
        local lst = clones.rnn[t]:forward(get_input(x, t, rnn_state[t-1]))
		
        rnn_state[t] = {}
        for i=1,#init_state do 
            table.insert(rnn_state[t], lst[i]) 
        end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        table.insert(rnn_state[t-1], drnn_state[t])
        local dlst = clones.rnn[t]:backward(get_input(x, t, rnn_state[t-1]), drnn_state[t])
        drnn_state[t-1] = {}
        local tmp = 1 -- not the safest way but quick
        for k,v in pairs(dlst) do
            if k > tmp then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-tmp] = v
            end
        end	
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    
    -- renormalize gradients
    local grad_norm, shrink_factor
    grad_norm = grad_params:norm()

    if grad_norm > opt.max_grad_norm then
        shrink_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(shrink_factor)
    end    

    return loss, grad_params
end

function load_network()
    print('load model from ' .. opt.model_dir)
    checkpoint = torch.load(opt.model_dir)

    protos = checkpoint.protos  
    checkpoint.opt.model_dir = opt.model_dir
    opt = checkpoint.opt
    epoch = checkpoint.epoch + 1 
    train_losses = checkpoint.train_losses
    val_loss = checkpoint.val_loss
    val_losses = checkpoint.val_losses
    i = checkpoint.i
    lr = checkpoint.lr
    set_optim(opt.optim_method, lr, opt.wdecay, opt.momentum)

    print("epoch: " .. epoch .. " train_losses: " .. train_losses[#train_losses] ) -- .. " val_loss: " .. val_losses[#val_losses])  
    checkpoint=nil  
    collectgarbage()
end

-- start optimization here
halfStep = 0
train_losses = {}
val_losses = {}
train_losses_epoch = {}
local tot_train_loss = 0
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
for i = 1, iterations do
    local epoch = i / loader.split_sizes[1]

    local timer = torch.Timer()
    local time = timer:time().real

    if opt.optim_method == 'sgd' then
    	_, loss = optim.sgd(feval, params, optim_state)
    elseif opt.optim_method == 'adam' then
    	_, loss = optim.adam(feval, params, optim_state)
    end

    local train_loss = torch.exp(loss[1])
    train_losses[i] = train_loss
    
    tot_train_loss = tot_train_loss + train_loss
    -- every now and then or on last iteration
    if i % loader.split_sizes[1] == 0 then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        train_losses_epoch[#train_losses_epoch+1] = tot_train_loss/loader.split_sizes[1]
        tot_train_loss = 0
        val_losses[#val_losses+1] = val_loss
        -- local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        local savefile = string.format('%s/lm_%s.t7', opt.checkpoint_dir, opt.savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.train_losses_epoch = train_losses_epoch
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.halfStep = halfStep
        checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
		checkpoint.lr = lr
        print(string.format("(epoch %.2f), lr=%.4f, train_loss = %6.4f, val_loss = %6.4f ", epoch,lr, train_losses_epoch[epoch], val_loss))
		if epoch == opt.max_epochs or lr <= 0.00005 or epoch == 20 then
            print('saving checkpoint to ' .. savefile)
            torch.save(savefile, checkpoint)
        end
    end
	 
    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), lr=%.4f, train_loss = %6.4f, grad/param norm = %6.4e", i, iterations, epoch,lr, train_loss, grad_params:norm() / params:norm()))
    end  
    -- decay learning rate after epoch
    if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
        if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
            lr = lr * opt.learning_rate_decay
            halfStep = halfStep + 1
            set_optim(opt.optim_method, lr, opt.wdecay, opt.momentum)
			end
    end    
 	
 	if lr <= 0.00005*opt.learning_rate_decay then
 		break
 	end

    if i % 10 == 0 then collectgarbage() end
    if opt.time ~= 0 then
       print("Batch Time:", timer:time().real - time)
    end
end

test_perp, token_perp, uniperp = eval_split(3)
print('Perplexity on test set: ' .. test_perp .. ', PPLu on test set: ' .. uniperp)
test_savefile = string.format('%s/lm_%s_epoch-test_%.2f.t7', opt.checkpoint_dir, opt.savefile, test_perp)
torch.save(test_savefile, {test_perp, token_perp, uniperp})

