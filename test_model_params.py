import torch
import sys
import os
import json

# 添加项目根目录到sys.path
project_root = '/data/home/star/IMSE'
sys.path.insert(0, project_root)

# 添加models目录到sys.path以便正确导入Multi_transformer
models_path = os.path.join(project_root, 'models')
sys.path.insert(0, models_path)

# 修复导入方式，使用绝对导入
from models.generator import IMSE  # 从generator.py文件导入IDSE类
from env import AttrDict  # 从env.py文件导入AttrDict类
from joblib import Parallel, delayed
import numpy as np


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):
    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)
    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_structure(model):
    """打印模型结构信息"""
    print("\n模型结构信息:")
    print("="*50)
    print(f"模型类名: {model.__class__.__name__}")
    print(f"模型文件: {model.__class__.__module__}")
    
    # 打印模型的主要组件
    print("\n模型主要组件:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
        
    # 打印模型的详细结构
    print("\n模型详细结构:")
    print(model)


def test_model_parameters():
    # 从config.json加载配置
    try:
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 转换为AttrDict对象
        h = AttrDict(config)
        
        # 显示配置信息
        print("配置信息:")
        print("="*30)
        for key, value in h.items():
            print(f"{key}: {value}")
        
        # 创建模型实例
        model = IMSE(h)
        
        # 计算参数量
        total_params = count_parameters(model)
        
        print(f"\n模型总参数量: {total_params:,}")
        print(f"模型总参数量 (百万): {total_params/1e6:.2f}M")
        
        # 显示模型结构
        print_model_structure(model)
        
        #显示各层参数量
        # print("\n各层参数量详情:")
        # print("="*30)
        # total = 0
        # for name, param in model.named_parameters():
        #     num_params = param.numel()
        #     total += num_params
        #     print(f"{name}: {num_params:,}")
        # print(f"总计: {total:,}")
            
    except FileNotFoundError as e:
        print(f"未找到config.json文件: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_parameters()