require 'lfs'
require 'torch'

data_root = 'data/ptb/'

dir_unigram_count = data_root  .. 'unigram.th7'
processed_file_name = data_root .. 'train.txt'

-- count unigram of training dataset
if not path.exists(dir_unigram_count) then
  unigram_count = {}
  num_sentence_train = 0
	num_sent = {}
  corpus_file_num = processed_file_name
  print("Processing file :"..processed_file_name)
  s = 0
  for sentence in io.lines(processed_file_name) do
    num_word =0 
    s = s+1
    for word in string.gmatch(sentence, "[^%s]+") do
      print(word)
      if unigram_count[word] == nil then
        unigram_count[word] = 1
      else
        unigram_count[word] = unigram_count[word] + 1
      end
      num_word = num_word+1
    end
    num_sentence_train=num_sentence_train+1
  end
  
  print("# train sentence : " .. num_sentence_train )
  print(unigram_count)
  torch.save(dir_unigram_count, unigram_count)
end


-- discarding all words with count below count_thre -> <unk>
processed_file_name_valid = data_root .. 'valid.txt'
processed_file_name_test = data_root .. 'test.txt'

count_thre = 1
processed_file_t7 = data_root .. 'data.t7'

if not path.exists(processed_file_t7) then
  eos = '<eos>'
  unk = '<unk>'
  processed_vocab_file = data_root .. 'vocab.t7'
  processed_freq_file = data_root .. 'freq.t7'

  unigram_count = torch.load(dir_unigram_count)
  num_w = 0
  nww = 0
  i2w = {}
  w2i = {}
  freq = {}
  for k, v in pairs(unigram_count) do
    if k ~= '<unk>' then
      num_w = num_w+1
      i2w[num_w] = k
      w2i[k] = num_w
      freq[num_w] = v
      if v >= count_thre then
        nww = nww+1
      end
    end
    print(num_w,k,v)
  end
  print('tot_words: ', nww)

  freq = torch.Tensor(freq)
  y, j = torch.sort(freq,1, true) -- decending order

  vocab = {}
  vocab.idx2word = {}
  vocab.word2idx = {}
  vocab.freq = {}
  vocab.freq[1] = 0
  vocab.idx2word[1] = unk
  vocab.word2idx[unk] = 1

  vocab.freq[2] = 0
  vocab.idx2word[2] = eos
  vocab.word2idx[eos] = 2

  for i = 3, nww+2 do
    id = j[i-2]
    vocab.idx2word[i] = i2w[id]
    vocab.word2idx[i2w[id]] = i
    vocab.freq[i] = freq[id]
  end

  torch.save(processed_vocab_file,vocab)
  torch.save(processed_freq_file,vocab.freq)

  outputall = {}
  outputall[1] = {}
  outputall[2] = {}
  outputall[3] = {}
  for i = 1, 3, 1 do
    sentence_length = 0
    num_unk = 0      
    num_words = 0
    local processed_txt
    if i==1 then
      print("Processing file train")
      corpus_file_num = processed_file_name
    elseif i==2 then
      print("Processing file valid")
      corpus_file_num = processed_file_name_valid         
    else
      print("Processing file test")
      corpus_file_num = processed_file_name_test
    end

    num_sent = 0
    for sentence in io.lines(corpus_file_num) do -- in corpus
      word_cur = 0
      for word in string.gmatch(sentence, "[^%s]+") do -- in sentence
        word_cur = word_cur + 1
        num_words = num_words + 1

        if vocab.word2idx[word] == nil or word == '<unk>' then
          word = unk
          num_unk = num_unk + 1
        end
        word_temp = word

        outputall[i][num_words] = vocab.word2idx[word_temp] --Train/Valid/Test
      end
      word_cur = word_cur + 1
      num_words = num_words + 1
      outputall[i][num_words] = vocab.word2idx[eos] --end of sentence
      num_sent = num_sent + 1
      sentence_length = sentence_length + word_cur
    end
    if i == 1 then
      vocab.freq[1] = num_unk
      vocab.freq[2] = num_sent
      torch.save(processed_vocab_file,vocab)
      torch.save(processed_freq_file,vocab.freq)
    end
    print("mean sentencee length = " .. sentence_length / num_sent)
    print("num sentence = ", num_sent)
    print("# <unk> = " .. num_unk .. " # words = " .. num_words .. " OOV rates = ".. num_unk / num_words * 100 .. "(%)")
  end
  outputall[1] = torch.Tensor(outputall[1]):long()
  outputall[2] = torch.Tensor(outputall[2]):long()
  outputall[3] = torch.Tensor(outputall[3]):long()
  torch.save(processed_file_t7,outputall)

end

