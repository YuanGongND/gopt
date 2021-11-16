# -*- coding: utf-8 -*-
# @Time    : 8/20/21 4:02 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : load_feats.py.py

# code modified from:

# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

import sys, os
import json
import numpy as np
import argparse
import kaldi_io
from utils import load_human_scores, load_phone_symbol_table

def get_args():
    parser = argparse.ArgumentParser(description='Extract GOP Feactures from the Kaldi so762 recipe', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feature_scp', default='exp/gop_train/feat.scp', help='Input gop-based feature file, in Kaldi scp')
    parser.add_argument('--phone-symbol-table', type=str, default='data/lang_nosp/phones-pure.txt', help='Phone symbol table')
    parser.add_argument('--human_scoring_json', default='data/local/scores.json', help='Input human scores file, in JSON format')
    parser.add_argument('--min-phone-idx', type=int, default=-1)
    parser.add_argument('--max-phone-idx', type=int, default=999)
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args

def main(args, set):

    if set == 'train':
        args.feature_scp = 'exp/gop_train/feat.scp'
    elif set == 'test':
        args.feature_scp = 'exp/gop_test/feat.scp'
    else:
        raise ValueError('set must be train or test')


    # Phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    with open(args.human_scoring_json) as f:
        info = json.loads(f.read())

    accuracy_of = {}
    stress_of = {}
    total_of = {}
    word_of = {}
    word_id_of = {}
    for utt in info:
        phone_num = 0
        for word_id, word in enumerate(info[utt]['words']):
            cur_word_text = info[utt]['words'][word_id]['text']
            cur_word_accuracy = info[utt]['words'][word_id]['accuracy']
            cur_word_stress = info[utt]['words'][word_id]['stress']
            cur_word_total = info[utt]['words'][word_id]['total']
            assert len(word['phones']) == len(word['phones-accuracy'])
            for i, phone in enumerate(word['phones']):
                key = f'{utt}.{phone_num}'
                phone_num += 1
                word_of[key] = cur_word_text
                word_id_of[key] = word_id
                accuracy_of[key] = cur_word_accuracy
                stress_of[key] = cur_word_stress
                total_of[key] = cur_word_total

    # Gather the features
    lables = []
    keys = []
    features = []
    cnt = 0
    for key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        # print(key)
        cnt += 1
        if key not in total_of:
            print(f'Warning: no human score for {key}')
            continue
        ph = int(feat[0])
        if ph in range(args.min_phone_idx, args.max_phone_idx + 1):
            if phone_int2sym is not None and ph in phone_int2sym:
                ph = phone_int2sym[ph]
            keys.append(key)
            features.append(feat)
            lables.append([ph, word_id_of[key], word_of[key], accuracy_of[key], stress_of[key], total_of[key]])

    print('now processing {:s} set, load {:d} samples'.format(set, cnt))

    if os.path.exists('gopt_feats') == False:
        os.mkdir('gopt_feats')

    if set == 'test':
        #np.savetxt('gopt_feats/te_feats_word.csv', features, delimiter=',')
        np.savetxt('gopt_feats/te_keys_word.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/te_labels_word.csv', lables, delimiter=',', fmt='%s')
    elif set == 'train':
        #np.savetxt('gopt_feats/yuan_tr_feats_word.csv', features, delimiter=',')
        np.savetxt('gopt_feats/tr_keys_word.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/tr_labels_word.csv', lables, delimiter=',', fmt='%s')

if __name__ == '__main__':
    args = get_args()
    main(args, 'train')
    main(args, 'test')
