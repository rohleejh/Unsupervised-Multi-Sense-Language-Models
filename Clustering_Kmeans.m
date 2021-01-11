%% Sentence clustering for each word
repeat = 10;
s = sprintf('%s/concat_sent_vec', stage2dir);
s1= sprintf('_lyr%d_hid%d%d.mat',num_layer, rnn_size, rnn_size);
ss = sprintf('_lyr%d_h%d%d_c%d.mat',num_layer, rnn_size, rnn_size, num_cluster);
load([s,s1]);
hidsize = size(sent_all,2);
load(sprintf('%s/num_sent_per_word.mat',stage2dir));
tot = 0;
index = zeros(size(sent_all,1),1);
centroid = zeros(size(num_sent_per_word,2)*num_cluster,hidsize);
for i = 1:size(num_sent_per_word,2)
    freq = num_sent_per_word(2,i);
    data = sent_all(tot+1:tot+freq,:);
    [IDX, C, SUMD, D] = kmeans(data, num_cluster,...
        'Replicates', repeat, 'Display','final');
    index(tot+1:tot+freq,1) = IDX;
    centroid((i-1)*num_cluster + 1: i*num_cluster,:) = C;
    tot = tot+freq;
    disp(i);
end

save(sprintf('%s/centroid%s',stage2dir,ss),'centroid');
save(sprintf('%s/index%s',stage2dir,ss),'index');