import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transformers import BertModel, BertTokenizer


# 1. 定义图像嵌入模型（修复维度崩溃 Bug + 最新权重 API）
class ImageEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(ImageEmbedding, self).__init__()
        # 使用最新的 torchvision 权重指定 API
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.feature_extractor(images)
        # 修正：显式从第 1 维展开，保留 Batch 维度 (N, 2048)
        features = torch.flatten(features, 1)
        embeddings = self.fc(features)
        # 规范化：对比学习通常需要对 Embedding 进行 L2 归一化
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# 2. 定义文本嵌入模型
class TextEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(TextEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.fc(outputs.pooler_output)
        # 规范化：L2 归一化
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# 3. 定义对比损失函数 (包含余弦相似度匹配)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings, labels):
        # 归一化后的向量点积即为余弦相似度
        cosine_sim = torch.sum(image_embeddings * text_embeddings, dim=1)

        # 正样本损失：相似度越接近 1 越好
        positive_loss = (1 - labels) * (1 - cosine_sim)
        # 负样本损失：相似度低于 margin 时损失为 0
        negative_loss = labels * torch.clamp(cosine_sim - self.margin, min=0)

        return positive_loss.mean() + negative_loss.mean()


# 4. 参数配置与模型初始化
embed_size = 256
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = ImageEmbedding(embed_size).to(device)
text_model = TextEmbedding(embed_size).to(device)
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(
    list(image_model.parameters()) + list(text_model.parameters()),
    lr=1e-4
)

# 5. 示例数据生成
dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dummy_texts = ["A picture of a cat"] * batch_size
tokens = tokenizer(
    dummy_texts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

# 标签：0 表示正样本（匹配），1 表示负样本（不匹配）
labels = torch.randint(0, 2, (batch_size,)).float().to(device)

# 6. 前向传播与优化步骤
image_embeddings = image_model(dummy_images)
text_embeddings = text_model(tokens['input_ids'], tokens['attention_mask'])
loss = loss_fn(image_embeddings, text_embeddings, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# 7. 输出测试
print("Image Embeddings Shape:", image_embeddings.shape)  # torch.Size([8, 256])
print("Text Embeddings Shape:", text_embeddings.shape)  # torch.Size([8, 256])
print("Loss Value:", loss.item())