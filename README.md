# Video-Captioner-Transformer
The code can be run on three video captioning datasets, including MSVD, MSR-VTT, and VATEX.
## Dependencies
- Python 3.7
- Pytorch 1.90
- numpy, scikit-image, h5py, requests

## Installation
First clone this repository

`git clone https://github.com/myccver/Clip4caption.git`

Then, please put [data](https://pan.baidu.com/s/1Ukd7zCFNR6_S0ruqHCNZJA?pwd=1234) and [coco_caption](https://pan.baidu.com/s/1f5QFk8fOlHNM7zJgFBw24w?pwd=1234 ) to the project.
## Results
The checkpoint can be found [weight](https://pan.baidu.com/s/1atc2IfFV9OPCFrU2CrLxwA?pwd=1234 )
| Dataset | BLEU@4 | METEOR | ROUGE-L | CIDEr | Checkpoint |
|-------|-------|-------|-------|-------|-------|
| MSVD | 59.5 | 39.6 | 76.3 | 110.4 | msvd_clip14.pth |
| MSR-VTT | 44.0 | 29.9 | 62.6 | 55.3 | msrvtt_clip14.pth |
| VATEX | 34.2 | 24.1 | 50.5 | 58.3 | vatex_clip14.pth |


## Training
### MSVD
```
python train.py --train_label_h5
data/metadata/msvd_train_sequencelabel.h5
--val_label_h5
data/metadata/msvd_val_sequencelabel.h5
--test_label_h5
data/metadata/msvd_test_sequencelabel.h5
--train_cocofmt_file
data/metadata/msvd_train_cocofmt.json
--val_cocofmt_file
data/metadata/msvd_val_cocofmt.json
--test_cocofmt_file
data/metadata/msvd_test_cocofmt.json
--train_bcmrscores_pkl
data/metadata/msvd_train_evalscores.pkl
--train_feat_h5
""
""
""
data/clip_feature/msvd_train_clip14_feats.h5
--val_feat_h5
""
""
""
data/clip_feature/msvd_val_clip14_feats.h5
--test_feat_h5
""
""
""
data/clip_feature/msvd_test_clip14_feats.h5
--beam_size
5
--rnn_size
768
--eval_metric
CIDEr
--language_eval
1
--max_epochs
30
--batch_size
128
--test_batch_size
64
--learning_rate
0.0001
--lr_update
20
--save_checkpoint_from
1
--num_chunks
1
--train_cached_tokens
data/metadata/msvd_train_ciderdf.pkl
--use_rl
0
--use_mixer
0
--mixer_from
-1
--use_it
0
--dr_baseline_captions
0
--dr_baseline_type
0
--loglevel
INFO
--use_eos
0
--model_file
output/CLIP14_XE_msvd/msvd_clip14.pth
--start_from
No
--result_file
output/CLIP14_XE_msvd/msvd_clip14.json
--vocab_embedding
data/origin_feature/update_msvd_vocab_whole_ViT-L-14_embedding.pkl
--input_encoding_size
768
--seed
1
```
### MSR-VTT
```
python train.py --train_label_h5
data/metadata/msrvtt_train_sequencelabel.h5
--val_label_h5
data/metadata/msrvtt_val_sequencelabel.h5
--test_label_h5
data/metadata/msrvtt_test_sequencelabel.h5
--train_cocofmt_file
data/metadata/msrvtt_train_cocofmt.json
--val_cocofmt_file
data/metadata/msrvtt_val_cocofmt.json
--test_cocofmt_file
data/metadata/msrvtt_test_cocofmt.json
--train_bcmrscores_pkl
data/metadata/msrvtt_train_evalscores.pkl
--train_feat_h5
""
""
""
data/clip_feature/msrvtt_train_clip14_feats.h5
--val_feat_h5
""
""
""
data/clip_feature/msrvtt_val_clip14_feats.h5
--test_feat_h5
""
""
""
data/clip_feature/msrvtt_test_clip14_feats.h5
--beam_size
5
--rnn_size
768
--eval_metric
CIDEr
--language_eval
1
--max_epochs
30
--batch_size
128
--test_batch_size
64
--learning_rate
0.0001
--lr_update
20
--save_checkpoint_from
1
--num_chunks
1
--train_cached_tokens
data/metadata/msrvtt_train_ciderdf.pkl
--use_rl
0
--use_mixer
0
--mixer_from
-1
--use_it
0
--dr_baseline_captions
0
--dr_baseline_type
0
--loglevel
INFO
--use_eos
0
--model_file
output/CLIP14_XE_msrvtt/msrvtt_clip14.pth
--start_from
No
--result_file
output/CLIP14_XE_msrvtt/msrvtt_clip14.json
--use_global_local_feature
1
--vocab_embedding
data/origin_feature/update_msrvtt_vocab_whole_ViT-L-14_embedding.pkl
--input_encoding_size
768
--seed
1
```
### VATEX
```
python train.py --train_label_h5
data/metadata/vatex_train_sequencelabel.h5
--val_label_h5
data/metadata/vatex_val_sequencelabel.h5
--test_label_h5
data/metadata/vatex_test_sequencelabel.h5
--train_cocofmt_file
data/metadata/vatex_train_cocofmt.json
--val_cocofmt_file
data/metadata/vatex_val_cocofmt.json
--test_cocofmt_file
data/metadata/vatex_test_cocofmt.json
--train_bcmrscores_pkl
data/metadata/vatex_train_evalscores.pkl
--train_feat_h5
""
""
""
data/clip_feature/vatex_train_clip14_feats.h5
--val_feat_h5
""
""
""
data/clip_feature/vatex_val_clip14_feats.h5
--test_feat_h5
""
""
""
data/clip_feature/vatex_test_clip14_feats.h5
--beam_size
5
--rnn_size
768
--eval_metric
CIDEr
--language_eval
1
--max_epochs
30
--batch_size
256
--test_batch_size
1024
--learning_rate
0.0002
--lr_update
20
--save_checkpoint_from
1
--num_chunks
1
--train_cached_tokens
data/metadata/vatex_train_ciderdf.pkl
--use_rl
0
--use_mixer
0
--mixer_from
-1
--use_it
0
--dr_baseline_captions
0
--dr_baseline_type
0
--loglevel
INFO
--use_eos
0
--model_file
output/CLIP14_XE_vatex/vatex_clip14.pth
--start_from
No
--result_file
output/CLIP14_XE_vatex/vatex_clip14.json
--use_global_local_feature
1
--vocab_embedding
data/origin_feature/update_vatex_vocab_whole_ViT-L-14_embedding.pkl
--input_encoding_size
768
--seed
1
```
## Test
### MSVD
```
python test.py --train_label_h5
data/metadata/msvd_train_sequencelabel.h5
--val_label_h5
data/metadata/msvd_val_sequencelabel.h5
--test_label_h5
data/metadata/msvd_test_sequencelabel.h5
--train_cocofmt_file
data/metadata/msvd_train_cocofmt.json
--val_cocofmt_file
data/metadata/msvd_val_cocofmt.json
--test_cocofmt_file
data/metadata/msvd_test_cocofmt.json
--train_bcmrscores_pkl
data/metadata/msvd_train_evalscores.pkl
--train_feat_h5
""
""
""
data/clip_feature/msvd_train_clip14_feats.h5
--val_feat_h5
""
""
""
data/clip_feature/msvd_val_clip14_feats.h5
--test_feat_h5
""
""
""
data/clip_feature/msvd_test_clip14_feats.h5
--beam_size
5
--rnn_size
768
--eval_metric
CIDEr
--language_eval
1
--max_epochs
30
--batch_size
128
--test_batch_size
64
--learning_rate
0.0001
--lr_update
20
--save_checkpoint_from
1
--num_chunks
1
--train_cached_tokens
data/metadata/msvd_train_ciderdf.pkl
--use_rl
0
--use_mixer
0
--mixer_from
-1
--use_it
0
--dr_baseline_captions
0
--dr_baseline_type
0
--loglevel
INFO
--use_eos
0
--model_file
output/CLIP14_XE_msvd/msvd_clip14.pth
--start_from
No
--result_file
output/CLIP14_XE_msvd/msvd_clip14.json
--vocab_embedding
data/origin_feature/update_msvd_vocab_whole_ViT-L-14_embedding.pkl
--input_encoding_size
768
--seed
1
```
