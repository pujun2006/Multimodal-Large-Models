import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from transformers import AutoTokenizer, AutoModel


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        emb = self.proj(feats)
        # 投影到单位超球面，使余弦相似度等价于内积，保证统一空间尺度
        return F.normalize(emb, p=2, dim=-1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.backbone(**encoded)

        # 使用 attention_mask 做均值池化，通常比单纯取 [CLS] 更稳健
        mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class SymmetricClipLoss(nn.Module):
    """双向对称对比损失（InfoNCE 风格），拉近正样本对、推开批内负样本对。"""

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature)))

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 限制缩放系数上限，防止训练早期梯度过大
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_image = logit_scale * (image_emb @ text_emb.t())
        logits_per_text = logits_per_image.t()

        batch_size = image_emb.size(0)
        labels = torch.arange(batch_size, device=image_emb.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        return loss, logits_per_image


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 512

    image_encoder = ImageEncoder(embedding_dim=embedding_dim, freeze_backbone=False).to(device)
    text_encoder = TextEncoder(embedding_dim=embedding_dim, freeze_backbone=False).to(device)
    criterion = SymmetricClipLoss(init_temperature=0.07).to(device)

    optimizer = optim.AdamW(
        [
            {"params": image_encoder.parameters(), "lr": 1e-4},
            {"params": text_encoder.parameters(), "lr": 1e-4},
            {"params": criterion.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-2,
    )

    # 构造一个批量大小为 4 的占位图文对批次（实际训练请替换为真实对齐数据集）
    # 图像张量已按 ImageNet 均值方差做过标准化，形状为 [B, 3, 224, 224]
    images = torch.randn(4, 3, 224, 224, device=device)
    texts = [
        "A photo of a dog playing with a ball.",
        "A cup of hot coffee on a wooden table.",
        "A sunset over a calm ocean horizon.",
        "A cyclist riding along a mountain road.",
    ]

    image_encoder.train()
    text_encoder.train()
    criterion.train()

    for step in range(5):
        optimizer.zero_grad()

        image_emb = image_encoder(images)
        text_emb = text_encoder(texts, device=device)
        loss, logits = criterion(image_emb, text_emb)

        loss.backward()
        optimizer.step()

        print(f"Step {step + 1} | Loss: {loss.item():.4f} | Max Cosine Sim: {(image_emb @ text_emb.t()).max().item():.4f}")

    # 推理阶段相似度矩阵示例（对角线为对应配对样本的余弦相似度）
    image_encoder.eval()
    text_encoder.eval()
    with torch.no_grad():
        query_img_emb = image_encoder(images)
        cand_txt_emb = text_encoder(texts, device=device)
        cosine_sim_matrix = query_img_emb @ cand_txt_emb.t()
        print("\n图文余弦相似度矩阵:\n", cosine_sim_matrix.cpu())


if __name__ == "__main__":
    main()
