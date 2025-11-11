# -*- coding: utf-8 -*-
# test_bev_shift.py
# 完整可运行的 BEV ego-motion shift 测试脚本
# 直接运行：python test_bev_shift.py

import numpy as np

# ========================================
# 1. 模拟数据：ohb_img_metas（来自 nuScenes 的 can_bus 格式）
# ========================================
# 假设 batch_size = 2
ohb_img_metas = [
    {
        # can_bus 共 18 个 float，关键是：
        # [0]: delta_x (m)   → 当前帧相对上一帧 X 平移
        # [1]: delta_y (m)   → Y 平移
        # [16]: ego yaw (rad) → 车头朝向（nuScenes 是第16个）
        'can_bus': [
            2.15,   # delta_x = 前进了 2.15 米
            0.83,   # delta_y = 右偏了 0.83 米
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5236, # ego yaw = 30° (rad) → 0.5236 rad
            0.0
        ]
    },
    {
        'can_bus': [
            -1.20,  # 后退了 1.2 米
            -0.50,  # 左偏了 0.5 米
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.7854,  # yaw = -45° → -π/4
            0.0
        ]
    }
]

# ========================================
# 2. BEV 配置（和 BEVFormer 官方完全一致）
# ========================================
bev_h = 200   # BEV 高度网格数
bev_w = 200   # BEV 宽度网格数

# ========================================
# 3. 你原来的代码（原封不动）
# ========================================
# obtain rotation angle and shift with ego motion
delta_x = np.array([each['can_bus'][0] for each in ohb_img_metas])
delta_y = np.array([each['can_bus'][1] for each in ohb_img_metas])
ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in ohb_img_metas])  # -2 就是第16个

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
real_w = pc_range[3] - pc_range[0]  # 102.4
real_h = pc_range[4] - pc_range[1]  # 102.4
grid_length = (real_h / bev_h, real_w / bev_w)
grid_length_y = grid_length[0]  # 0.512 m/grid
grid_length_x = grid_length[1]  # 0.512 m/grid

translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
bev_angle = ego_angle - translation_angle

shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

use_shift = True
shift_y = shift_y * use_shift
shift_x = shift_x * use_shift

# ========================================
# 4. 打印结果（方便你看效果）
# ========================================
print("=== BEV Ego-Motion Shift 计算结果 ===")
print(f"delta_x       : {delta_x}")
print(f"delta_y       : {delta_y}")
print(f"ego_angle (°) : {ego_angle}")
print(f"translation_length (m): {translation_length}")
print(f"translation_angle (°): {translation_angle}")
print(f"bev_angle (°): {bev_angle}")
print(f"grid_length_y (m/grid): {grid_length_y:.4f}")
print(f"grid_length_x (m/grid): {grid_length_x:.4f}")
print(f"→ shift_y (normalized): {shift_y}")
print(f"→ shift_x (normalized): {shift_x}")
print("\n这些 shift_y, shift_x 可以直接喂给 BEVFormer 的 temporal alignment 模块！")

# ========================================
# 5. 验证：转成 PyTorch 张量（真实使用方式）
# ========================================
import torch
shift = torch.stack([torch.from_numpy(shift_x), torch.from_numpy(shift_y)], dim=-1)
print(f"\nPyTorch shift tensor: {shift.shape} → {shift}")