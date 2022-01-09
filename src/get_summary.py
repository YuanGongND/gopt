# -*- coding: utf-8 -*-
# @Time    : 9/21/21 9:50 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_summary.py

import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

requirement_list = {'gopt': 0.0, 'lstm': 0.0}

# second pass
for requirement in requirement_list.keys():
    threshold = requirement_list[requirement]
    result = []
    root_path = '../exp/'
    exp_list = get_immediate_subdirectories(root_path)
    exp_list.sort()
    for exp in exp_list:
        if requirement in exp and os.path.isfile(root_path + exp + '/result_summary.csv'):
            try:
                print(exp)
                cur_res = np.loadtxt(root_path + exp + '/result_summary.csv', delimiter=',')[0]
                print(cur_res)
                test_mse = cur_res[2]
                test_corr = cur_res[3]
                te_utt_corr = cur_res[20:25]
                te_word_corr = cur_res[28:31]

                print(te_utt_corr)
                result.append([exp, test_mse, test_corr, te_utt_corr[4], te_word_corr[2]])
            except:
                pass

    np.savetxt('../exp/' + requirement + '_summary_brief.csv', result, delimiter=',', fmt='%s')