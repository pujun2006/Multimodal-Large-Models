import torch
import torch.nn as nn
import torch.optim as optim


# 1. 修正后的共享注意力模块 (显式声明 batch_first=True)
class SharedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(SharedAttention, self).__init__()
        # 必须指定 batch_first=True 以匹配 (B, Seq_Len, Dim) 形状
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, 2, hidden_dim)
        attn_output, _ = self.attention(x, x, x)
        # 残差连接 + LayerNorm
        out = self.norm(x + attn_output)
        return self.relu(out)


# 2. 任务专属头部
class TaskHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(TaskHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 3. 多任务共享参数模型
class SharedParameterModel(nn.Module):
    def __init__(self, img_input_dim, text_input_dim, hidden_dim, img_output_dim, text_output_dim):
        super(SharedParameterModel, self).__init__()
        self.img_encoder = nn.Linear(img_input_dim, hidden_dim)
        self.text_encoder = nn.Linear(text_input_dim, hidden_dim)

        # 共享注意力层
        self.shared_attention = SharedAttention(hidden_dim=hidden_dim, num_heads=4)

        # 任务专属 Head
        self.img_task_head = TaskHead(hidden_dim, img_output_dim)
        self.text_task_head = TaskHead(hidden_dim, text_output_dim)

    def forward(self, img_input, text_input):
        # 1. 图像和文本独立映射到同一隐空间 [B, hidden_dim]
        img_features = self.img_encoder(img_input)
        text_features = self.text_encoder(text_input)

        # 2. 堆叠为序列形式: [B, 2, hidden_dim] (Index 0: Image, Index 1: Text)
        combined_features = torch.stack((img_features, text_features), dim=1)

        # 3. 通过共享 Cross-Modal Attention 交互
        shared_features = self.shared_attention(combined_features)  # [B, 2, hidden_dim]

        # 4. 按索引解包并过任务 Head
        img_output = self.img_task_head(shared_features[:, 0, :])  # [B, img_output_dim]
        text_output = self.text_task_head(shared_features[:, 1, :])  # [B, text_output_dim]
        return img_output, text_output


# 4. 超参数与设备选择
img_input_dim = 256
text_input_dim = 128
hidden_dim = 64
img_output_dim = 10  # 图像 10 分类
text_output_dim = 2  # 文本 2 分类
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 模型初始化
model = SharedParameterModel(img_input_dim, text_input_dim, hidden_dim, img_output_dim, text_output_dim).to(device)
img_criterion = nn.CrossEntropyLoss()
text_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. 模拟数据
img_data = torch.randn(batch_size, img_input_dim).to(device)
text_data = torch.randn(batch_size, text_input_dim).to(device)
img_labels = torch.randint(0, img_output_dim, (batch_size,)).to(device)
text_labels = torch.randint(0, text_output_dim, (batch_size,)).to(device)

# 7. 训练过程
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    img_outputs, text_outputs = model(img_data, text_data)

    img_loss = img_criterion(img_outputs, img_labels)
    text_loss = text_criterion(text_outputs, text_labels)
    total_loss = img_loss + text_loss

    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Image Loss: {img_loss.item():.4f}, Text Loss: {text_loss.item():.4f}")

# 8. 模型评估
model.eval()
test_img_data = torch.randn(batch_size, img_input_dim).to(device)
test_text_data = torch.randn(batch_size, text_input_dim).to(device)

with torch.no_grad():
    test_img_outputs, test_text_outputs = model(test_img_data, test_text_data)
    img_predictions = torch.argmax(test_img_outputs, dim=1)
    text_predictions = torch.argmax(test_text_outputs, dim=1)

print("\nImage Predictions:", img_predictions)
print("Text Predictions:", text_predictions)