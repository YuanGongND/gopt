# -*- coding: utf-8 -*-
# @Time    : 9/19/21 11:13 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_seq_data_phn.py

# Generate sequence phone input and label for seq2seq models from raw Kaldi GOP features.

import numpy as np
import json

def load_feat(path):
    file = np.loadtxt(path, delimiter=',')
    return file

def load_keys(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def load_label(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def process_label(label):
    pure_label = []
    for i in range(0, label.shape[0]):
        pure_label.append(float(label[i, 1]))
    return np.array(pure_label)

def process_feat_seq_word(feat, keys, labels):
    key_set = []
    for i in range(keys.shape[0]):
        cur_key = keys[i].split('.')[0]
        key_set.append(cur_key)

    utt_cnt = len(list(set(key_set)))
    print('In total utterance number : ' + str(utt_cnt))

    # -1 means n/a
    seq_label = np.zeros([utt_cnt, 50, 4]) - 1

    prev_utt_id = keys[0].split('.')[0]

    row = 0
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id

        seq_label[row, cur_tok_id, 0:3] = labels[i, 3:6]
        seq_label[row, cur_tok_id, 3] = labels[i, 1]

    return seq_label

# utt label dict
with open('scores.json') as f:
    utt2score = json.loads(f.read())

# sequence training data
tr_feat = load_feat('../../data/raw_kaldi_gop/librispeech/tr_feats.csv')
tr_keys = load_keys('../../data/raw_kaldi_gop/librispeech/tr_keys_word.csv')
tr_label = load_label('../../data/raw_kaldi_gop/librispeech/tr_labels_word.csv')
tr_label = process_feat_seq_word(tr_feat, tr_keys, tr_label)
print(tr_label.shape)
np.save('../../data/seq_data_librispeech/tr_label_word.npy', tr_label)

# sequence test data
te_feat = load_feat('../../data/raw_kaldi_gop/librispeech/te_feats.csv')
te_keys = load_keys('../../data/raw_kaldi_gop/librispeech/te_keys_word.csv')
te_label = load_label('../../data/raw_kaldi_gop/librispeech/te_labels_word.csv')
te_label = process_feat_seq_word(te_feat, te_keys, te_label)
print(te_label.shape)
np.save('../../data/seq_data_librispeech/te_label_word.npy', te_label)
