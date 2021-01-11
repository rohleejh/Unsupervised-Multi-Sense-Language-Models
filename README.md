# Unsupervised-Multi-Sense-Language-Models
Unsupervised Multi-Sense Language Models for Natural Language Processing Tasks

## Requirements
Code is tested on Torch7(lua) and Matlab 2017a. 
Torch7 and its library can be installed via:
```
sh insall_torch7_lib.sh
```

## Data
Data should be put into the `data/` directory, split into `train.txt`, `valid.txt`, and `test.txt`
Please generate tensor file before training SSLM(baseline) via:
```
th make_text_to_tensor.lua
```

## Stage 1
SSLM--LSTM with 650 hidden vectors
```
sh run_models_baseline.sh
```

## Stage 2
1) Extract context vector from SSLM--LSTM models.
2) K-means clustering on context vectors with K=3
3) Re-index single-sense tokens to multi-sense tokens
(Context vectors and centroids of clusters are saved in `stage2_file_ptb/`)

This can be performed via the following script:
```
sh run_Stage2.sh
```

## Stage 3
MSLM--LSTM with 650 hidden vectors and a cetain number of clusters
```
sh run_MSLM_LSTM.sh
```
