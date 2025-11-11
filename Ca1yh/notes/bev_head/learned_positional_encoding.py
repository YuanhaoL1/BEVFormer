# ==================== debug_learned_pos.py ====================
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, num_feats=4, row_num_embed=10, col_num_embed=15):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats

        # 初始化让值好看（方便你看规律）
        with torch.no_grad():
            self.row_embed.weight.data = torch.arange(row_num_embed).float(
            ).unsqueeze(1) * 10 + torch.arange(num_feats)
            self.col_embed.weight.data = torch.arange(col_num_embed).float(
            ).unsqueeze(1) * 100 + torch.arange(num_feats)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)

        x_embed = self.col_embed(x)  # [w, C]
        y_embed = self.row_embed(y)  # [h, C]


        print(x_embed.unsqueeze(0).shape)
        print(y_embed.unsqueeze(1).shape)
        # Step 1: 构建 [h, w, C]
        x_part = x_embed.unsqueeze(0).repeat(h, 1, 1)  # [h, w, C]
        y_part = y_embed.unsqueeze(1).repeat(1, w, 1)  # [h, w, C]

        # Step 2: 拼接 → [h, w, 2*C]
        pos = torch.cat([x_part, y_part], dim=-1)  # [h, w, 2*C]  ← 3维！

        # Step 3: 正确转置 + 加 batch
        pos = pos.permute(2, 0, 1)  # [2*C, h, w]     ← 正确！
        pos = pos.unsqueeze(0)  # [1, 2*C, h, w]
        pos = pos.repeat(mask.shape[0], 1, 1, 1)  # [bs, 2*C, h, w]

        return pos

# ========================== 一键运行测试 ==========================
if __name__ == "__main__":
    # 创建模型（小尺寸方便看清）
    pos_encoder = LearnedPositionalEncoding(
        num_feats=4,        # 方便看数字
        row_num_embed=5,
        col_num_embed=6
    )

    # 模拟输入
    bs, h, w = 2, 5, 6
    mask = torch.zeros(bs, h, w, dtype=torch.bool)
    mask[1, 3:, 4:] = True  # 随便加点 padding

    # 运行 forward（会自动打印每一步！）
    pos = pos_encoder(mask)

    print(f"\n最终输出 pos.shape: {pos.shape}")
    print("你现在可以手动查看 pos[0, :, 0, 0] 等任意位置！")
