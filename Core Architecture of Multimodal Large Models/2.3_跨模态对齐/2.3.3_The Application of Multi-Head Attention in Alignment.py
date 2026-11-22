import torch
import torch.nn as nn
import torch.optim as optim

# 1. 多头注意力机制（增加 Mask 支持）
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]
        Q = self.query(queries)
        K = self.key(keys)
        V = self.value(values)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        return self.fc_out(out)

# 2. 图文对齐模型（修复维度对齐与池化）
class CrossModalAlignment(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossModalAlignment, self).__init__()
        self.image_proj = nn.Linear(embed_size, embed_size)
        self.text_proj = nn.Linear(embed_size, embed_size)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, image_embeddings, text_embeddings):
        # 1. 投影层
        img_feats = self.image_proj(image_embeddings)  # [B, img_len, embed_size]
        txt_feats = self.text_proj(text_embeddings)    # [B, txt_len, embed_size]

        # 2. 交叉注意力：用图像 Query 去关注文本 K/V
        combined = self.attention(queries=img_feats, keys=txt_feats, values=txt_feats) # [B, img_len, embed_size]

        # 3. 序列维度池化聚合：将 (B, img_len, embed_size) 压缩为全局表示 (B, embed_size)
        pooled_feats = combined.mean(dim=1) # [B, embed_size]

        # 4. 预测输出：压缩为一维分类 Logits (B,)
        logits = self.fc(pooled_feats).squeeze(-1) # [B]
        return logits

# 3. 数据与超参数
embed_size = 256
num_heads = 8
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 初始化模型与优化器
model = CrossModalAlignment(embed_size, num_heads).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 5. 示例输入数据
image_embeddings = torch.randn(batch_size, 10, embed_size).to(device)  # 10 个图像区域
text_embeddings = torch.randn(batch_size, 15, embed_size).to(device)   # 15 个文本 Token
labels = torch.randint(0, 2, (batch_size,)).float().to(device)           # 图文匹配标签 (0/1)

# 6. 前向传播与损失计算
outputs = model(image_embeddings, text_embeddings) # 形状: [8]
loss = criterion(outputs, labels)

# 7. 反向传播与优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 8. 输出测试
print("Output Shape:", outputs.shape)  # 正确输出: torch.Size([8])
print("Loss Value:", loss.item())