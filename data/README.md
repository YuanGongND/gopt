If you want to skip Kaldi GOP recipe and data preprocessing, you can use our intermediate files and directly go to step 3.
This is useful if you do not want to change the acoustic model and save some time. It also makes it easy to fairly compare your own model with our GOPT model. 

Download from this [dropbox link](https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1) or [腾讯微云链接](https://share.weiyun.com/vJCAXjFY), extract from the compressed package, and place the folds in this directory in the following format.

```
data
│   README.md
└───raw_kaldi_gop
│   │   librispeech
│   │   │   tr_feats.csv
│   │   │   tr_keys_phn.csv
│   │   │   tr_keys_word.csv
│   │   │   tr_feats.csv
│   │   │   ... (10 files in total)
│   
└───seq_data_librispeech
    │   tr_feat.npy
    │   tr_label_phn.npy
    │   tr_label_word.npy
    │   tr_label_utt.npy
    │   ... (8 files in toal)
└───seq_data_paiia
    │   tr_feat.npy
    │   tr_label_phn.npy
    │   tr_label_word.npy
    │   tr_label_utt.npy
    │   ... (8 files in toal)
└───seq_data_paiib
    │   tr_feat.npy
    │   tr_label_phn.npy
    │   tr_label_word.npy
    │   tr_label_utt.npy
    │   ... (8 files in toal)
```