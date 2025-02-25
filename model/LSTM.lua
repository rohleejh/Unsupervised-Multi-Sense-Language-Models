local LSTM = {}

local ok, cunn = pcall(require, 'fbcunn')
LookupTable = nn.LookupTable

function LSTM.lstm(rnn_size, n, dropout,word_vocab_size, word_vec_size, batch_norm)
	-- delete char_vocab_size, char_vec_size, feature_maps, kernels, length, use_words, use_chars, highway_layers, hsm 
    -- rnn_size = dimensionality of hidden layers
    -- n = number of layers
    -- dropout = dropout probability
    -- word_vocab_size = num words in the vocab    
    -- word_vec_size = dimensionality of word embeddings
  
    dropout = dropout or 0 
    
    -- there will be 2*n+1 inputs if using words or chars, 
	-- otherwise there will be 2*n + 2 inputs   
	local word_vec_layer, x, input_size_L, word_vec
    -- local char_vec_layer, word_vec_layer, x, input_size_L, word_vec, char_vec

	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- batch_size x 1 (word indices)
	word_vec_layer = LookupTable(word_vocab_size, word_vec_size)
	word_vec_layer.name = 'word_vecs' -- change name so we can refer to it easily later

    for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end
    local outputs = {}
    for L = 1,n do
    	-- c,h from previous timesteps. 
		local prev_h = inputs[L*2+1]
		local prev_c = inputs[L*2]
		-- the input to this layer
		if L == 1 then
			x = word_vec_layer(inputs[1])
			input_size_L = word_vec_size
			if batch_norm == 1 then	
				x = nn.BatchNormalization(0)(x)
			end
		else 
			x = outputs[(L-1)*2] -- prev_h
			if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
			input_size_L = rnn_size
		end
		-- evaluate the input sums at once for efficiency
		local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
		local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
		local all_input_sums = nn.CAddTable()({i2h, h2h})
		
		local sigmoid_chunk = nn.Narrow(2, 1, 3*rnn_size)(all_input_sums)
		sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
		local in_gate = nn.Narrow(2,1,rnn_size)(sigmoid_chunk)
		local out_gate = nn.Narrow(2, rnn_size+1, rnn_size)(sigmoid_chunk)
		local forget_gate = nn.Narrow(2, 2*rnn_size + 1, rnn_size)(sigmoid_chunk)
		local in_transform = nn.Tanh()(nn.Narrow(2,3*rnn_size + 1, rnn_size)(all_input_sums))

		-- perform the LSTM update
		local next_c = nn.CAddTable()({
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate, in_transform})
		})
		-- gated cells form the output
		local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
    end

 	-- set up the decoder
    local top_h = outputs[#outputs]
    if dropout > 0 then 
        top_h = nn.Dropout(dropout)(top_h) 
    else
        top_h = nn.Identity()(top_h) --to be compatiable with dropout=0 
	end
	
	local proj = nn.Linear(rnn_size, word_vocab_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)
    return nn.gModule(inputs, outputs)
end

return LSTM

