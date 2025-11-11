# ==================== 一键运行：看穿多头注意力每一步形状 ====================
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 实现 transpose_qkv 和 transpose_output（原书函数）
def transpose_qkv(X, num_heads):
    """为了多头注意力计算而变换形状"""
    # 输入 X: [bs, seq_len, num_hiddens]
    # 输出: [bs*num_heads, seq_len, num_hiddens/num_heads]
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)  # [bs, num_heads, seq_len, head_dim]
    return X.reshape(-1, X.shape[2], X.shape[3])  # [bs*num_heads, seq_len, head_dim]

def transpose_output(X, num_heads):
    """逆转 transpose_qkv 的输出"""
    # 输入 X: [bs*num_heads, seq_len, head_dim]
    # 输出: [bs, seq_len, num_hiddens]
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# 2. 简化的 DotProductAttention（只支持缩放点积）
class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (d ** 0.5)  # [bs*h, seq_q, seq_k]
        print(f"    scores shape: {scores.shape}")
        
        if valid_lens is not None:
            mask = torch.arange(scores.shape[-1], device=scores.device).unsqueeze(0).unsqueeze(0)
            mask = mask >= valid_lens.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        print(f"    attention_weights shape: {attention_weights.shape}")
        
        output = torch.bmm(attention_weights, values)
        print(f"    single-head output shape: {output.shape}")
        return output

# 3. 完整多头注意力模块（带 print 每一步！）
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        print(f"\n输入 shapes:")
        print(f"  queries:  {queries.shape}")
        print(f"  keys:     {keys.shape}")
        print(f"  values:   {values.shape}")
        if valid_lens is not None:
            print(f"  valid_lens: {valid_lens}")

        # Step 1: 线性投影
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)
        print(f"\nStep 1: 线性投影后")
        print(f"  Q: {Q.shape}")
        print(f"  K: {K.shape}")
        print(f"  V: {V.shape}")

        # Step 2: transpose_qkv → 分头
        queries = transpose_qkv(Q, self.num_heads)
        keys = transpose_qkv(K, self.num_heads)
        values = transpose_qkv(V, self.num_heads)
        print(f"\nStep 2: transpose_qkv 后（分头）")
        print(f"  queries: {queries.shape}  ← bs*{self.num_heads}, seq, {Q.shape[-1]//self.num_heads}")
        print(f"  keys:    {keys.shape}")
        print(f"  values:  {values.shape}")

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            print(f"  valid_lens 扩展后: {valid_lens.shape}")

        # Step 3: 缩放点积注意力
        print(f"\nStep 3: 进入注意力计算...")
        output = self.attention(queries, keys, values, valid_lens)

        # Step 4: transpose_output → 合并头
        print(f"\nStep 4: transpose_output 前: {output.shape}")
        output_concat = transpose_output(output, self.num_heads)
        print(f"  transpose_output 后: {output_concat.shape}")

        # Step 5: 输出投影
        output = self.W_o(output_concat)
        print(f"\nStep 5: W_o 输出投影后: {output.shape}")

        return output

# ==================== 一键运行测试 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len_q = 5
    seq_len_kv = 7
    num_hiddens = 64
    num_heads = 8
    dropout = 0.5

    # 随机输入
    queries = torch.randn(batch_size, seq_len_q, num_hiddens)
    keys = torch.randn(batch_size, seq_len_kv, num_hiddens)
    values = torch.randn(batch_size, seq_len_kv, num_hiddens)
    valid_lens = torch.tensor([4, 2])  # 第一个序列有效长度4，第二个只有2

    # 创建模型
    attn = MultiHeadAttention(
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        num_hiddens=num_hiddens,
        num_heads=num_heads,
        dropout=dropout
    )

    # eval 模式（关闭 dropout 方便看）
    attn.eval()
    with torch.no_grad():
        output = attn(queries, keys, values, valid_lens)

    print(f"\n最终输出 shape: {output.shape}")
    print("多头注意力 forward 每一步 shape 全打印完毕！")