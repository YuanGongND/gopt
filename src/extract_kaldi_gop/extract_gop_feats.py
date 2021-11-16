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
    parser.add_argument('--floor', type=float, default=0.1)
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

    # Human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=args.floor)

    # Gather the features
    lables = []
    keys = []
    features = []
    cnt = 0
    for key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        cnt += 1
        if key not in score_of:
            print(f'Warning: no human score for {key}')
            continue
        ph = int(feat[0])
        if ph in range(args.min_phone_idx, args.max_phone_idx + 1):
            if phone_int2sym is not None and ph in phone_int2sym:
                ph = phone_int2sym[ph]
            keys.append(key)
            features.append(feat)
            lables.append([ph, score_of[key]])

    print('now processing {:s} set with floor {:f}, load {:d} samples'.format(set, args.floor, cnt))

    if os.path.exists('gopt_feats') == False:
        os.mkdir('gopt_feats')

    if set == 'test':
        np.savetxt('gopt_feats/te_feats.csv', features, delimiter=',')
        np.savetxt('gopt_feats/te_keys_phn.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/te_labels_phn.csv', lables, delimiter=',', fmt='%s')
    elif set == 'train':
        np.savetxt('gopt_feats/tr_feats_phn.csv', features, delimiter=',')
        np.savetxt('gopt_feats/tr_keys_phn.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/tr_labels_phn.csv', lables, delimiter=',', fmt='%s')

if __name__ == '__main__':
    args = get_args()
    args.floor = 0.1
    main(args, 'train')

    args.floor = 0.1
    main(args, 'test')
