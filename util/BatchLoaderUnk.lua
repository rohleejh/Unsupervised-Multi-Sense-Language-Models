
-- Modified from https://github.com/yoonkim/lstm-char-cnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoaderUnk = {}
local stringx = require('pl.stringx')
BatchLoaderUnk.__index = BatchLoaderUnk
utf8 = require 'lua-utf8'

function BatchLoaderUnk.create(data_dir,batch_size, seq_length, exp_ver)
    local self = {}
    setmetatable(self, BatchLoaderUnk)

    local vocab_file = path.join(data_dir, 'vocab.t7')
	local tensor_file = path.join(data_dir, 'data.t7')
	if opt.num_cluster > 1 then
		local file_name = string.format('data_c%d.t7',opt.num_cluster)
		tensor_file = path.join(data_dir, file_name)
		file_name = string.format('vocab_c%d.t7',opt.num_cluster)
		vocab_file = path.join(data_dir, file_name)
	end
	if exp_ver == 'sil' or exp_ver == 'gap' then
		local file_name = string.format('data_%s.t7',opt.num_cluster, exp_ver)
		tensor_file = path.join(data_dir, file_name)
		file_name = string.format('vocab_%s.t7',opt.num_cluster, exp_ver)
		vocab_file = path.join(data_dir, file_name)
	end

    -- construct a tensor with all the data
	if not (path.exists(tensor_file)) then
		print('Please compile "make_text_to_tensor.lua" ')
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
	local v = torch.load(vocab_file)
	self.idx2word = v.idx2word
	self.word2idx = v.word2idx
    self.vocab_size = #self.idx2word
    
	print('make unigrams ... ')
	_vocabsize = torch.max(all_data[1])
    local unigram
	local unigram_temp
	if opt.num_cluster == 1 then
		unigram_file = path.join(data_dir, 'unigram_org.t7')
	end
	if opt.num_cluster > 1 then
		unigram_temp = string.format('unigram_c%d.t7', opt.num_cluster)
		if exp_ver == 'sil' or exp_ver == 'gap' then
			unigram_temp = string.format('unigram_%s.t7', exp_ver)
		end
		unigram_file = path.join(data_dir, unigram_temp)
	end

	print('unigram_file: ', unigram_file)
	if not (path.exists(unigram_file)) then
		total_words = all_data[1]:size(1)
		unigram = torch.zeros(_vocabsize)
		for i = 1, all_data[1]:size(1) do
			idx = all_data[1][i]
			if i%5000000 == 0 then
				print(i)
			end
			unigram[idx] = unigram[idx] + 1
		end
		unigram = torch.div(unigram,total_words)
		torch.save(unigram_file, unigram)
	else
		unigram = torch.load(unigram_file)
	end
	self.unigram = unigram

    if opt.num_cluster > 1 then 
    	self.idx2orgidx = v.idx2orgidx
    	self.org2stridx = v.org2stridx
    end
    
    print(string.format('Word vocab size: %d', self.vocab_size))
    -- cut off the end for train/valid sets so that it divides evenly
    -- test set is not cut off
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.split_sizes = {}
    self.all_batches = {}
    print('reshaping tensors...')  
    local x_batches, y_batches, nbatches
    for split, data in ipairs(all_data) do
       local len = data:size(1)
       if len % (batch_size * seq_length) ~= 0 and split < 3 then
          data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
       end
       
       local ydata = data:clone()
       ydata:sub(1,-2):copy(data:sub(2,-1))
       ydata[-1] = data[1]
       local ynew = torch.Tensor()

       if split < 3 then
          x_batches = data:view(batch_size, -1):split(seq_length, 2)
          y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
          nbatches = #x_batches	   
          self.split_sizes[split] = nbatches
          assert(#x_batches == #y_batches)
       else --for test we repeat dimensions to batch size (easier but inefficient evaluation)
          x_batches = {data:resize(1, data:size(1)):expand(batch_size, data:size(2))}
          y_batches = {ydata:resize(1, ydata:size(1)):expand(batch_size, ydata:size(2))}
          self.split_sizes[split] = 1	
       end
	   self.all_batches[split] = {x_batches, y_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', 
          self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoaderUnk:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoaderUnk:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
	return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx]
end

return BatchLoaderUnk

