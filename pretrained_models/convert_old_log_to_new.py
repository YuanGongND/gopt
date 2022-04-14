# -*- coding: utf-8 -*-
# @Time    : 4/14/22 2:00 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : convert_old_log_to_new.py

# Note: the logs we uploaded are produced by an old script, it does not have column `0` (epoch) and does not have a header, new scripts adds the epoch info and header.
# This script convert it to new style with epoch info and header

import numpy as np

def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header

def convert_log(old_path):
    old_log = np.loadtxt(old_path, delimiter=',')
    # save the old
    np.savetxt(old_path[:-4]+'_old.csv', old_log, delimiter=',')
    new_log = np.zeros([100, 32])
    for i in range(100):
        new_log[i, 0] = i
    new_log[:, 1:] = old_log
    header = ','.join(gen_result_header())
    np.savetxt(old_path, new_log, delimiter=',', header=header)

convert_log('/Users/yuan/Documents/gopt/pretrained_models/gopt_paiib/result.csv')