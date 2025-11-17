#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="make test file list")

# 我们只需要提供一个包含所有测试文件的目录路径
# 我会使用你之前脚本中的 "noisy_test" 路径作为默认值
parser.add_argument('--test_dir_to_scan', type=str, 
                    default='/data/home/star/MUSE-Speech-Enhancement/VB_DEMAND_16K/noisy_test', 
                    help='Path to a test directory to scan for filenames (e.g., noisy_test or clean_test)')

args = parser.parse_args()

print(f"Scanning directory: {args.test_dir_to_scan}")

try:
    # 1. 扫描你指定的目录，只获取.wav文件
    test_file_names = sorted([f for f in os.listdir(args.test_dir_to_scan) if f.endswith('.wav')])
except FileNotFoundError:
    print(f"Error: Directory not found at {args.test_dir_to_scan}")
    print("Please check the --test_dir_to_scan path.")
    exit()

if not test_file_names:
    print(f"Warning: No .wav files found in {args.test_dir_to_scan}")
else:
    print(f"Found {len(test_file_names)} .wav files. Generating test.txt...")

# 2. 写入 test.txt
with open('test.txt', 'w') as test_txt:
    for file_name in tqdm(test_file_names):
        # 提取文件名 (不带.wav)
        file_id = file_name.split('.')[0]
        
        # 写入文件ID。cal_metrics.py 会正确读取这个ID。
        test_txt.write(file_id + '|\n')

print("Successfully created test_final.txt.")