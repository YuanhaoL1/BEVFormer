import torch
import torch.nn as nn

# 1. 创建
embed = nn.Embedding(num_embeddings=9000, embedding_dim=256)  # BEVFormer 的 bev_queries
print(embed.weight.shape)   # [9000, 256]
# 2. 输入：整数索引
indices = torch.arange(0, 9000)           # [0,1,2,...,8999]
indices = indices.unsqueeze(0).repeat(2, 1)  # [2, 9000] → bs=2

# # 3. 输出：向量
# output = embed(indices)                   # [2, 9000, 256]
# print(output.shape)   # [2, 9000, 256]
# print(output.dtype)  # float32


single = torch.tensor(9001)

out = embed(single)

print(out.shape)   # [1, 256]
print(out.dtype)    # float32