# test_stack_reshape.py
import torch

# ================== 1. 模拟 BEVFormer 中的输入 ==================
print("=== 1. 创建模拟的 query (当前帧 BEV 特征) ===")
bs = 2          # batch size = 2
len_bev = 4     # BEV 网格数，比如 2x2 展平后是 4（实际是 200x200=40000）
c = 3           # 通道数，简化成 3

# 随机生成当前帧的 query，方便看出数据变化
query = torch.arange(1, bs * len_bev * c + 1).float().reshape(bs, len_bev, c)
query = query + 100 * torch.arange(bs).unsqueeze(1).unsqueeze(2)  # 让不同 batch 区别明显
print("query.shape:", query.shape)        # [2, 4, 3]
print("query = \n", query)
print()

# ================== 2. torch.stack([query, query], dim=1) ==================
print("=== 2. stack 两个相同的 query，模拟「当前帧 + 历史帧」 ===")
stacked = torch.stack([query, query], dim=1)   # dim=1 表示在时间维度堆叠
print("stacked.shape:", stacked.shape)        # [2, 2, 4, 3]
print("stacked = \n", stacked)
print("可以看到：第0维是 batch，第1维是时间（0=当前帧，1=历史帧，但内容一样）")
print()

# ================== 3. reshape(bs*2, len_bev, c) ==================
print("=== 3. reshape 成 (bs*2, len_bev, c) ===")
reshaped = stacked.reshape(bs * 2, len_bev, c)
print("reshaped.shape:", reshaped.shape)      # [4, 4, 3]
print("reshaped = \n", reshaped)
print()

# ================== 4. 看清楚每一块是谁 ==================
print("=== 4. 手动拆开看：前 bs 行是当前帧，后 bs 行是历史帧 ===")
print("前 bs=2 个样本（当前帧）：")
print(reshaped[:bs])
print()
print("后 bs=2 个样本（历史帧，内容和当前帧一样）：")
print(reshaped[bs:])
print()

# ================== 5. 为什么 Transformer 喜欢这种形状？ ==================
print("=== 5. 为什么这么 reshape？因为 MultiheadAttention 要 batch_first ===")
print("现在 reshaped 看起来就像有 4 个样本：")
print("  [0]: batch0 的当前帧")
print("  [1]: batch1 的当前帧")
print("  [2]: batch0 的历史帧")
print("  [3]: batch1 的历史帧")
print("所以注意力机制会自动做「当前帧 ↔ 历史帧」跨时间交互！")
print()

# ================== 6. 验证和原始 stack 是否一致 ==================
print("=== 6. 验证数据没变，只是视角变了 ===")
print("stacked[0,0] (batch0 当前帧): \n", stacked[0, 0])
print("reshaped[0] (对应位置): \n", reshaped[0])
print("stacked[0,1] (batch0 历史帧): \n", stacked[0, 1])
print("reshaped[2] (对应位置): \n", reshaped[2])
print("完全一样！说明 reshape 只是换了个“看数据的角度”")

# ================== 7. 完整一行代码版 ==================
print("\n=== 完整一行代码等价写法 ===")
value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)
print("value.shape:", value.shape)
print("value = \n", value)