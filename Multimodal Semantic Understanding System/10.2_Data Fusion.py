from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer


class FeatureExtractor(nn.Module):
    """封装 BERT 文本特征提取与 ResNet50 图像特征提取"""

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        freeze_backbones: bool = True,
        max_length: int = 128,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.text_encoder = BertModel.from_pretrained(bert_name)

        # 使用 ResNet50 池化层输出（2048 维），去掉默认 1000 类分类头
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        self.text_dim = self.text_encoder.config.hidden_size  # 768
        self.image_dim = 2048  # ResNet50 avgpool 输出维度

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(**encoded)
        token_embeddings = outputs.last_hidden_state  # [B, L, 768]
        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        # 仅对有效 token 做平均池化，排除 [PAD]
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [B, 768]

    def encode_image(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        rgb_images = [img.convert("RGB") for img in images]
        batch_tensor = torch.stack(
            [self.image_transform(img) for img in rgb_images], dim=0
        ).to(device)
        return self.image_encoder(batch_tensor)  # [B, 2048]


class ModalFusion(nn.Module):
    """特征级多模态融合模块（支持交叉注意力或自注意力 + 残差归一化）"""

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fusion_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.GELU(),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.GELU(),
        )

        # 文本 -> 图像交叉注意力（查询来自文本，键值来自图像）
        self.t2i_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 图像 -> 文本交叉注意力（查询来自图像，键值来自文本）
        self.i2t_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 可选：双模态序列自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        # [B, text_dim] -> [B, 1, fusion_dim]
        t_proj = self.text_projection(text_features).unsqueeze(1)
        # [B, image_dim] -> [B, 1, fusion_dim]
        i_proj = self.image_projection(image_features).unsqueeze(1)

        if self.use_cross_attention:
            t_attended, _ = self.t2i_attn(query=t_proj, key=i_proj, value=i_proj)
            i_attended, _ = self.i2t_attn(query=i_proj, key=t_proj, value=t_proj)

            t_out = self.norm1(t_proj + t_attended)
            i_out = self.norm2(i_proj + i_attended)
            joint_repr = 0.5 * (t_out.squeeze(1) + i_out.squeeze(1))  # [B, fusion_dim]
        else:
            # [B, 2, fusion_dim]
            combined = torch.cat([t_proj, i_proj], dim=1)
            attn_out, _ = self.self_attn(combined, combined, combined)
            joint_repr = self.norm1(combined + attn_out).mean(dim=1)  # [B, fusion_dim]

        return self.output_layer(joint_repr)


class MultimodalClassifier(nn.Module):
    """端到端融合模型：特征提取 -> 特征级融合 -> 下游任务分类头"""

    def __init__(
        self,
        num_classes: int,
        fusion_dim: int = 512,
        freeze_backbones: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.extractor = FeatureExtractor(freeze_backbones=freeze_backbones)
        self.fusion = ModalFusion(
            text_dim=self.extractor.text_dim,
            image_dim=self.extractor.image_dim,
            fusion_dim=fusion_dim,
            use_cross_attention=use_cross_attention,
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        texts: List[str],
        images: List[Image.Image],
        backbone_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.fusion.parameters()).device

        # 冻结骨干或推理/微调控制时可按需关闭骨干梯度
        ctx = torch.enable_grad() if backbone_grad else torch.no_grad()
        with ctx:
            text_feats = self.extractor.encode_text(texts, device=device)
            image_feats = self.extractor.encode_image(images, device=device)

        # 融合与分类头始终保留梯度以便训练
        fused_features = self.fusion(text_feats.detach(), image_feats.detach())
        logits = self.classifier(fused_features)
        return fused_features, logits


def _load_or_create_dummy_images(paths: List[str]) -> List[Image.Image]:
    """若本地不存在示例图片，则生成临时纯色图片方便直接运行"""
    fallback_colors = [(220, 120, 90), (90, 160, 220)]
    loaded: List[Image.Image] = []
    for p, color in zip(paths, fallback_colors):
        path = Path(p)
        if not path.exists():
            Image.new("RGB", (300, 300), color=color).save(path)
        with Image.open(path) as img:
            loaded.append(img.convert("RGB"))
    return loaded


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = [
        "A cat sitting on a mat.",
        "A dog running in the park.",
    ]
    image_paths = ["cat.jpg", "dog.jpg"]
    images = _load_or_create_dummy_images(image_paths)

    # 假设做 2 类下游分类演示（如猫 vs 狗）
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    model = MultimodalClassifier(
        num_classes=2,
        fusion_dim=512,
        freeze_backbones=True,
        use_cross_attention=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    model.train()
    optimizer.zero_grad()

    fused_features, logits = model(texts, images, backbone_grad=False)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    print("融合后特征形状:", fused_features.shape)  # [2, 512]
    print("分类输出 Logits 形状:", logits.shape)  # [2, 2]
    print(f"训练一步后交叉熵损失: {loss.item():.4f}")