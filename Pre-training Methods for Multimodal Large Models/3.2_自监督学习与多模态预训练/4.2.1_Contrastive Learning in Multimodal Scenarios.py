import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer


class ImageTextPairDataset(Dataset):
    """图文配对数据集示例（演示用随机图像 + 不同描述文本，替换为你自己的真实数据即可）"""

    def __init__(self, num_samples: int = 160, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size
        # 为每个样本构造独立文本，避免批内文本完全相同造成假负样本冲突
        self.captions = [
            f"A sample photograph #{i} showing a distinct object or scene."
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # 真实场景请替换为 PIL.Image 读取 + torchvision.transforms 预处理与归一化
        image = torch.rand(3, self.image_size, self.image_size)
        caption = self.captions[idx]
        return image, caption


def build_collate_fn(tokenizer, max_length: int = 128):
    """在 DataLoader 阶段批量完成文本分词，提高训练吞吐"""

    def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images, texts = zip(*batch)
        stacked_images = torch.stack(images, dim=0)
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return stacked_images, encoded

    return collate_fn


class ImageEncoder(nn.Module):
    """基于 ResNet-50 的图像特征提取器"""

    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 去掉最后的分类全连接层，保留全局池化后的 2048 维特征
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return features.flatten(1)  # [B, 2048]


class TextEncoder(nn.Module):
    """基于预训练 Transformer 的文本特征提取器"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 位置隐藏状态作为句向量表征 [B, hidden_size]
        return output.last_hidden_state[:, 0, :]


class MultimodalContrastiveModel(nn.Module):
    """多模态对比学习双塔模型（图文共享嵌入空间 + 双向相似度 Logits）"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        image_dim: int = 2048,
        text_dim: int = 768,
        projection_dim: int = 256,
        init_tau: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(model_name=model_name)

        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # 可学习的温度缩放参数：logit_scale = log(1 / tau)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_tau))

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # L2 归一化后映射到单位超球面
        img_embeddings = F.normalize(self.image_projector(img_features), dim=-1)
        text_embeddings = F.normalize(self.text_projector(text_features), dim=-1)

        # 限制缩放系数上限，避免训练中出现梯度尖峰
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        logits_per_image = logit_scale * (img_embeddings @ text_embeddings.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> torch.Tensor:
    """双向 InfoNCE / 交叉熵对比损失（Image->Text 与 Text->Image 对称）"""
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def train(
    model_name: str = "bert-base-uncased",
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-4,
    max_length: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = ImageTextPairDataset(num_samples=160, image_size=224)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # 保证每个 batch 大小一致，便于对比学习矩阵运算
        collate_fn=build_collate_fn(tokenizer, max_length=max_length),
    )

    model = MultimodalContrastiveModel(
        model_name=model_name,
        image_dim=2048,
        text_dim=768,
        projection_dim=256,
        init_tau=0.07,
    ).to(device)

    # 图像编码器、文本编码器与投影头可按需设置不同学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (images, encoded_texts) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            input_ids = encoded_texts["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded_texts["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()

            logits_per_image, logits_per_text = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 5 == 0:
                current_tau = 1.0 / model.logit_scale.exp().clamp(max=100.0).item()
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | Step [{step + 1}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} | Tau: {current_tau:.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch + 1} 完成 | 平均 Loss: {avg_loss:.4f} ===")

    print("模型训练完成，多模态对比学习任务成功实现。")


if __name__ == "__main__":
    train()