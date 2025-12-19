#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 RCBANet 模型的 FLOPs 和参数量
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgl_rpcanet.models.rcbanet import RCBANet, RCBANetConfig


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    # 创建模型配置（与 train.py 当前配置相同）
    cfg = RCBANetConfig(
        C1=256, C2=256, C3=256,  # 当前配置
        cau_ch_s1=128, cau_ch_s2=128, cau_ch_s3=128,  # 当前配置
        cau_kernels=[3, 5],
        cau_strides=[8, 8, 8],  # 当前配置
        base_ch_s1=32, base_ch_s2=32, base_ch_s3=24,
        mid_ch_s1=24, mid_ch_s2=24, mid_ch_s3=16,
        prox_depth=2,
        grad_depth=2,
        use_gradient_checkpointing=False,  # 计算FLOPs时关闭
        deep_supervision=True,
    )
    
    model = RCBANet(cfg)
    model.eval()
    
    # 输入尺寸
    input_size = (1, 1, 256, 256)  # Batch=1, Channel=1, H=256, W=256
    dummy_input = torch.randn(input_size)
    
    print("=" * 60)
    print("RCBANet 模型复杂度分析")
    print("=" * 60)
    print(f"\n输入尺寸: {input_size}")
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 尝试使用 thop 计算 FLOPs
    try:
        from thop import profile, clever_format
        
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
        flops_formatted = clever_format([macs * 2], "%.3f")[0]
        
        print(f"\n--- 使用 thop 计算 ---")
        print(f"MACs (乘加运算): {macs_formatted}")
        print(f"FLOPs (约等于 2×MACs): {flops_formatted}")
        print(f"  原始值: {macs * 2 / 1e9:.3f} GFLOPs")
        print(f"参数量: {params_formatted}")
        
    except ImportError:
        print("\n[提示] 未安装 thop 库，请运行: pip install thop")
    
    # 尝试使用 ptflops 计算 FLOPs
    try:
        from ptflops import get_model_complexity_info
        
        macs, params = get_model_complexity_info(
            model, 
            (1, 256, 256),  # 输入通道数, H, W
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        
        print(f"\n--- 使用 ptflops 计算 ---")
        print(f"MACs: {macs / 1e9:.3f} GMACs")
        print(f"FLOPs: {macs * 2 / 1e9:.3f} GFLOPs")
        print(f"参数量: {params / 1e6:.3f} M")
        
    except ImportError:
        print("\n[提示] 未安装 ptflops 库，请运行: pip install ptflops")
    
    # 尝试使用 fvcore 计算 FLOPs
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        flops = FlopCountAnalysis(model, dummy_input)
        
        print(f"\n--- 使用 fvcore 计算 ---")
        print(f"FLOPs: {flops.total() / 1e9:.3f} GFLOPs")
        
        # 可选：打印各层详细信息
        # print("\n各模块 FLOPs 分布:")
        # print(flop_count_table(flops, max_depth=3))
        
    except ImportError:
        print("\n[提示] 未安装 fvcore 库，请运行: pip install fvcore")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

