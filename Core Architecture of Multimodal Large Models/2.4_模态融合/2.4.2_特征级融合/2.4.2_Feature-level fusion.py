import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定义多模态特征级融合模型 (Feature-Level Fusion Model)
class FeatureFusionModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(FeatureFusionModel, self).__init__()
        # 单模态编码器
        self.text_encoder = nn.Linear(text_input_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_input_dim, hidden_dim)

        # 融合与分类头
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text_input, image_input):
        # 1. 独立特征编码
        text_features = self.dropout(self.relu(self.text_encoder(text_input)))
        image_features = self.dropout(self.relu(self.image_encoder(image_input)))

        # 2. 特征级拼接 (Feature Concat)
        fused_features = torch.cat((text_features, image_features), dim=1)

        # 3. 融合网络与分类
        fused_output = self.dropout(self.relu(self.fusion_layer(fused_features)))
        logits = self.classifier(fused_output)
        return logits


# 2. 超参数与设备选择
batch_size = 16
text_input_dim = 128
image_input_dim = 256
hidden_dim = 64
output_dim = 2  # 情感分类：0 (负面) 与 1 (正面)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 模型实例化与优化器
model = FeatureFusionModel(text_input_dim, image_input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. 模拟输入数据
text_data = torch.randn(batch_size, text_input_dim).to(device)
image_data = torch.randn(batch_size, image_input_dim).to(device)
labels = torch.randint(0, 2, (batch_size,)).to(device)

# 5. 训练过程
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(text_data, image_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 6. 模型评估 (加上 torch.no_grad() 节省显存)
model.eval()
test_text_data = torch.randn(batch_size, text_input_dim).to(device)
test_image_data = torch.randn(batch_size, image_input_dim).to(device)

with torch.no_grad():
    test_outputs = model(test_text_data, test_image_data)
    predictions = torch.argmax(test_outputs, dim=1)

print("\nPredictions:", predictions)