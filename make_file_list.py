#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description = "make file list")
parser.add_argument('--train_clean_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/clean_train/', help = 'train clean path')
parser.add_argument('--train_noisy_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/noisy_train/', help = 'train noisy path')
parser.add_argument('--test_clean_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/clean_test/', help = 'test clean path')
parser.add_argument('--test_noisy_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/noisy_test/', help = 'test noisy path')
# parser.add_argument('--test_clean_final_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/generated_files/MUSE-Next_pesq_3.368', help = 'test clean final path')
# parser.add_argument('--test_noisy_final_path', type = str, default = '/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/noisy_test', help = 'test noisy final path')

args = parser.parse_args()

train_clean_path = args.train_clean_path
train_noisy_path = args.train_noisy_path
test_clean_path = args.test_clean_path
test_noisy_path = args.test_noisy_path
# test_clean_final_path = args.test_clean_final_path
# test_noisy_final_path = args.test_noisy_final_path

train_file_names = sorted(os.listdir(train_clean_path))
test_file_names = sorted(os.listdir(test_clean_path))

with open('training.txt', 'w') as train_txt:
    for file_name in tqdm(train_file_names):
        train_txt.write(file_name.split('.')[0] + '|' + os.path.join(train_clean_path, file_name) + '|' + os.path.join(train_noisy_path, file_name) + '\n')

with open('test.txt', 'w') as test_txt:
    for file_name in tqdm(test_file_names):
        test_txt.write(file_name.split('.')[0] + '|' + os.path.join(test_clean_path, file_name) + '|' + os.path.join(test_noisy_path, file_name) + '\n')
