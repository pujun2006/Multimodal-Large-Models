from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer


class MultimodalDataset(Dataset):
    """支持同时加载文本与图像的多模态数据集（含基础校验与容错）"""

    def __init__(
        self,
        texts: List[str],
        image_paths: List[str],
        image_transform: Optional[transforms.Compose] = None,
    ):
        if not texts or not image_paths:
            raise ValueError("文本列表和图像路径列表均不能为空。")
        if len(texts) != len(image_paths):
            raise ValueError(
                f"文本数量 ({len(texts)}) 与图像数量 ({len(image_paths)}) 必须一致。"
            )

        self.texts = texts
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.texts)

    def _load_image(self, path_str: str) -> torch.Tensor:
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"找不到图像文件: {path}")
        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                return self.image_transform(rgb_img)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"图像读取或解码失败 ({path}): {exc}") from exc

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        text = str(self.texts[idx]).strip()
        if not text:
            raise ValueError(f"第 {idx} 条文本为空。")
        image_tensor = self._load_image(self.image_paths[idx])
        return text, image_tensor


class MultimodalAligner(nn.Module):
    """基于 BERT + ResNet50 的图文特征提取与共享空间对齐模块"""

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        proj_dim: int = 512,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.text_encoder = BertModel.from_pretrained(bert_name)

        # 使用 ResNet50 倒数第二层的 2048 维池化特征（去掉 1000 类分类头）
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        text_dim = self.text_encoder.config.hidden_size  # 768
        image_dim = 2048

        # 将不同模态特征投影到同一共享向量空间并做 L2 归一化
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # 可学习的温度系数（用于对比学习）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        if freeze_backbones:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        # 使用 attention_mask 做掩码平均池化，避免 [PAD] 干扰语义向量
        outputs = self.text_encoder(**encoded)
        token_embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts

        text_emb = self.text_proj(pooled)
        return F.normalize(text_emb, p=2, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(images)  # [B, 2048]
        img_emb = self.image_proj(img_feat)    # [B, proj_dim]
        return F.normalize(img_emb, p=2, dim=-1)

    def forward(
        self, texts: List[str], images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = images.device
        text_embeds = self.encode_text(texts, device=device)
        image_embeds = self.encode_image(images)

        # 计算图文余弦相似度矩阵（缩放后作为 logits）
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_text = logit_scale * text_embeds @ image_embeds.t()
        logits_per_image = logits_per_text.t()
        return text_embeds, image_embeds, logits_per_text


def contrastive_alignment_loss(logits_per_text: torch.Tensor) -> torch.Tensor:
    """对称对比损失（InfoNCE）：拉近配对图文，推开非配对图文"""
    batch_size = logits_per_text.size(0)
    targets = torch.arange(batch_size, device=logits_per_text.device)
    logits_per_image = logits_per_text.t()

    loss_t2i = F.cross_entropy(logits_per_text, targets)
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    return 0.5 * (loss_t2i + loss_i2t)


def collate_multimodal_batch(
    batch: List[Tuple[str, torch.Tensor]],
) -> Tuple[List[str], torch.Tensor]:
    texts, image_tensors = zip(*batch)
    return list(texts), torch.stack(image_tensors, dim=0)


def _create_dummy_images_if_missing(paths: List[str]) -> None:
    """仅用于本地无图片时快速跑通示例"""
    colors = [(220, 120, 90), (90, 160, 220)]
    for p, color in zip(paths, colors):
        path = Path(p)
        if not path.exists():
            Image.new("RGB", (300, 300), color=color).save(path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_texts = [
        "A cat sitting on a mat.",
        "A dog playing in the park.",
    ]
    sample_image_paths = ["cat.png", "dog.png"]
    _create_dummy_images_if_missing(sample_image_paths)

    # 1. 数据输入模块：Dataset + DataLoader
    dataset = MultimodalDataset(texts=sample_texts, image_paths=sample_image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
    )

    # 2. 初始化多模态特征提取与对齐模型
    model = MultimodalAligner(
        bert_name="bert-base-uncased",
        proj_dim=512,
        freeze_backbones=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    # 3. 演示一步前向对齐与对比损失计算
    model.train()
    for batch_texts, batch_images in dataloader:
        batch_images = batch_images.to(device)

        optimizer.zero_grad()
        text_embeds, image_embeds, logits_per_text = model(batch_texts, batch_images)
        loss = contrastive_alignment_loss(logits_per_text)
        loss.backward()
        optimizer.step()

        print(f"文本嵌入形状: {text_embeds.shape}")
        print(f"图像嵌入形状: {image_embeds.shape}")
        print(f"图文相似度矩阵形状: {logits_per_text.shape}")
        print(f"对齐对比损失: {loss.item():.4f}")