# -*- coding: utf-8 -*-
# @Time    : 9/19/21 11:13 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_seq_data_phn.py

# Generate sequence phone input and label for seq2seq models from raw Kaldi GOP features.

import numpy as np

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

def process_feat_seq(feat, keys, labels, phn_dict):
    key_set = []
    for i in range(keys.shape[0]):
        cur_key = keys[i].split('.')[0]
        key_set.append(cur_key)

    feat_dim = feat.shape[1] - 1

    utt_cnt = len(list(set(key_set)))
    print('In total utterance number : ' + str(utt_cnt))

    # Pad all sequence to 50 because the longest sequence of the so762 dataset is shorter than 50.
    seq_feat = np.zeros([utt_cnt, 50, feat_dim])
    # -1 means n/a, padded token
    # [utt, seq_len, 0] is the phone label, and the [utt, seq_len, 1] is the score label
    seq_label = np.zeros([utt_cnt, 50, 2]) - 1

    # the key format is utt_id.phn_id
    prev_utt_id = keys[0].split('.')[0]

    row = 0
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        # if a new sequence, start a new row of the feature vector.
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id

        # The first element is the phone label.
        seq_feat[row, cur_tok_id, :] = feat[i, 1:]

        # [utt, seq_len, 0] is the phone label
        seq_label[row, cur_tok_id, 0] = phn_dict[labels[i, 0]]
        # [utt, seq_len, 1] is the score label, range from 0-2
        seq_label[row, cur_tok_id, 1] = labels[i, 1]

    return seq_feat, seq_label

def gen_phn_dict(label):
    phn_dict = {}
    phn_idx = 0
    for i in range(label.shape[0]):
        if label[i, 0] not in phn_dict:
            phn_dict[label[i, 0]] = phn_idx
            phn_idx += 1
    return phn_dict

# generate sequence training data
tr_feat = load_feat('../../data/raw_kaldi_gop/librispeech/tr_feats.csv')
tr_keys = load_keys('../../data/raw_kaldi_gop/librispeech/tr_keys_phn.csv')
tr_label = load_label('../../data/raw_kaldi_gop/librispeech/tr_labels_phn.csv')
phn_dict = gen_phn_dict(tr_label)
print(phn_dict)
tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict)
print(tr_feat.shape)
print(tr_label.shape)
np.save('../../data/seq_data_librispeech/tr_feat.npy', tr_feat)
np.save('../../data/seq_data_librispeech/tr_label_phn.npy', tr_label)

# generate sequence test data
te_feat = load_feat('../../data/raw_kaldi_gop/librispeech/te_feats.csv')
te_keys = load_keys('../../data/raw_kaldi_gop/librispeech/te_keys_phn.csv')
te_label = load_label('../../data/raw_kaldi_gop/librispeech/te_labels_phn.csv')
te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict)
print(te_feat.shape)
print(te_label.shape)
np.save('../../data/seq_data_librispeech/te_feat.npy', te_feat)
np.save('../../data/seq_data_librispeech/te_label_phn.npy', te_label)
