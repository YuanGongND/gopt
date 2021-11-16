# -*- coding: utf-8 -*-
# @Time    : 9/23/21 11:33 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : collect_summary.py

# collect summery of repeated experiment.

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./test", help="directory to dump experiments")
args = parser.parse_args()

result = []
# for each repeat experiment
for i in range(0, 10):
    cur_exp_dir = args.exp_dir + '-' + str(i)
    if os.path.isfile(cur_exp_dir + '/result.csv'):
        try:
            print(cur_exp_dir)
            cur_res = np.loadtxt(cur_exp_dir + '/result.csv', delimiter=',')
            for last_epoch in range(cur_res.shape[0] - 1, -1, -1):
                if cur_res[last_epoch, 0] > 5e-03:
                    break
            result.append(cur_res[last_epoch, :])
        except:
            pass

result = np.array(result)
# get mean / std of the repeat experiments.
result_mean = np.mean(result, axis=0)
result_std = np.std(result, axis=0)
if os.path.exists(args.exp_dir) == False:
    os.mkdir(args.exp_dir)
np.savetxt(args.exp_dir + '/result_summary.csv', [result_mean, result_std], delimiter=',')