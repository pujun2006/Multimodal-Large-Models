import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer


class ImageTextPairDataset(Dataset):
    """对齐的图文对数据集占位实现（实际使用请替换为真实图片路径与描述）"""

    def __init__(self, num_samples: int = 16, transform: transforms.Compose | None = None):
        self.num_samples = num_samples
        self.transform = transform
        sample_texts = [
            "A cat sitting on a couch.",
            "A dog playing fetch in the park.",
            "A cup of hot coffee on a wooden table.",
            "A sunset over a calm ocean horizon.",
        ]
        self.texts = [sample_texts[i % len(sample_texts)] for i in range(num_samples)]
        # 生成 [H, W, C] uint8 张量以模拟真实 PIL/ndarray 图像输入
        self.images = [
            torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.texts[idx]


class ImageEncoder(nn.Module):
    """图像编码器：输出 L2 归一化的共享空间单位向量"""

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
        return F.normalize(emb, p=2, dim=-1)


class TextEncoder(nn.Module):
    """文本编码器：基于 attention_mask 均值池化并输出 L2 归一化单位向量"""

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

        mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class SymmetricContrastiveLoss(nn.Module):
    """全可导双向对称对比损失（InfoNCE），支持批内多负样本对比"""

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature))
        )

    def forward(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 单位向量内积即为余弦相似度，范围 [-1, 1]
        cosine_sim = image_emb @ text_emb.t()
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_image = logit_scale * cosine_sim
        logits_per_text = logits_per_image.t()

        batch_size = image_emb.size(0)
        labels = torch.arange(batch_size, device=image_emb.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        return loss, cosine_sim


def compute_recall_at_k(sim_matrix: torch.Tensor, k: int = 1) -> tuple[float, float]:
    """计算批量对角线对齐下的文搜图（T2I）与图搜文（I2T）Recall@K"""
    num_items = sim_matrix.size(0)
    k = min(k, num_items)
    targets = torch.arange(num_items, device=sim_matrix.device)

    # 图像查询检索文本（行方向取 top-k）
    i2t_topk = sim_matrix.topk(k, dim=1).indices
    i2t_recall = (i2t_topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    # 文本查询检索图像（列方向取 top-k）
    t2i_topk = sim_matrix.t().topk(k, dim=1).indices
    t2i_recall = (t2i_topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return i2t_recall, t2i_recall


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 512
    batch_size = 8

    # 与 ResNet50_Weights.DEFAULT 匹配的预处理管线
    img_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageTextPairDataset(num_samples=16, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    image_encoder = ImageEncoder(embedding_dim=embedding_dim, freeze_backbone=False).to(device)
    text_encoder = TextEncoder(embedding_dim=embedding_dim, freeze_backbone=False).to(device)
    criterion = SymmetricContrastiveLoss(init_temperature=0.07).to(device)

    optimizer = optim.AdamW(
        [
            {"params": image_encoder.parameters(), "lr": 1e-4},
            {"params": text_encoder.parameters(), "lr": 1e-4},
            {"params": criterion.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-2,
    )

    image_encoder.train()
    text_encoder.train()
    criterion.train()

    for epoch in range(3):
        for step, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            text_list = list(texts)

            optimizer.zero_grad()
            image_emb = image_encoder(images)
            text_emb = text_encoder(text_list, device=device)
            loss, cosine_sim = criterion(image_emb, text_emb)

            loss.backward()
            optimizer.step()

            if (step + 1) % 2 == 0 or step == len(dataloader) - 1:
                i2t_r1, t2i_r1 = compute_recall_at_k(cosine_sim.detach(), k=1)
                print(
                    f"Epoch {epoch + 1} | Step {step + 1} | "
                    f"Loss: {loss.item():.4f} | I2T R@1: {i2t_r1:.2f} | T2I R@1: {t2i_r1:.2f}"
                )

    # 推理阶段：计算余弦相似度矩阵并演示检索排序
    image_encoder.eval()
    text_encoder.eval()
    eval_images, eval_texts = next(iter(dataloader))
    eval_images = eval_images.to(device)
    eval_text_list = list(eval_texts)

    with torch.no_grad():
        img_emb = image_encoder(eval_images)
        txt_emb = text_encoder(eval_text_list, device=device)
        sim_matrix = img_emb @ txt_emb.t()
        print("\n余弦相似度矩阵形状:", tuple(sim_matrix.shape))
        print("对角线配对均值相似度:", sim_matrix.diag().mean().item())


if __name__ == "__main__":
    main()