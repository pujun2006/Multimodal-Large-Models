import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer


class VideoEncoder(nn.Module):
    """多帧视觉编码器：逐帧提取空间特征 + 时序 Transformer 建模 + 投影映射"""

    def __init__(self, embedding_dim: int = 512, max_frames: int = 32, dropout: float = 0.1):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.frame_backbone = backbone

        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, max_frames, feat_dim))
        nn.init.normal_(self.temporal_pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=8,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim, bias=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        返回 L2 归一化的视频嵌入: (B, D)
        """
        bsz, num_frames, channels, height, width = frames.shape
        if num_frames > self.temporal_pos_emb.size(1):
            raise ValueError(
                f"输入帧数 {num_frames} 超过 max_frames={self.temporal_pos_emb.size(1)}"
            )

        flat_frames = frames.view(bsz * num_frames, channels, height, width)
        frame_feats = self.frame_backbone(flat_frames)  # (B*T, feat_dim)
        seq_feats = frame_feats.view(bsz, num_frames, -1)  # (B, T, feat_dim)

        seq_feats = seq_feats + self.temporal_pos_emb[:, :num_frames, :]
        temporal_feats = self.temporal_encoder(seq_feats)  # (B, T, feat_dim)

        video_feat = temporal_feats.mean(dim=1)  # (B, feat_dim)
        video_emb = self.proj(video_feat)  # (B, D)
        return F.normalize(video_emb, dim=-1)


class TextEncoder(nn.Module):
    """文本编码器：预训练语言模型 + 投影映射"""

    def __init__(self, model_name: str = "bert-base-uncased", embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim, bias=False),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids, attention_mask: (B, L)
        返回 L2 归一化的文本嵌入: (B, D)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 向量（也可替换为 attention_mask 加权平均池化）
        cls_feat = outputs.last_hidden_state[:, 0, :]
        text_emb = self.proj(cls_feat)  # (B, D)
        return F.normalize(text_emb, dim=-1)


class SymmetricInfoNCELoss(nn.Module):
    """双向对称 InfoNCE 对比损失，支持可学习温度系数"""

    def __init__(self, init_temperature: float = 0.07, max_logit_scale: float = 100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def forward(self, video_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        video_emb: (B, D) 已 L2 归一化
        text_emb:  (B, D) 已 L2 归一化
        """
        scale = torch.clamp(self.logit_scale.exp(), max=self.max_logit_scale)
        logits_v2t = scale * torch.matmul(video_emb, text_emb.t())  # (B, B)，第 i 行对应视频 i 对各文本
        logits_t2v = logits_v2t.t()  # (B, B)，第 j 行对应文本 j 对各视频

        targets = torch.arange(video_emb.size(0), device=video_emb.device)
        loss_v2t = F.cross_entropy(logits_v2t, targets)
        loss_t2v = F.cross_entropy(logits_t2v, targets)
        return 0.5 * (loss_v2t + loss_t2v)


def build_frame_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_dummy_batch(
    tokenizer: AutoTokenizer,
    frame_transform: transforms.Compose,
    batch_size: int = 8,
    num_frames: int = 8,
    device: torch.device = torch.device("cpu"),
):
    """构造多样本示例批次：(B, T, C, H, W) 视频张量与对应文本编码"""
    raw_frames = torch.rand(batch_size, num_frames, 3, 224, 224)
    flat = raw_frames.view(batch_size * num_frames, 3, 224, 224)
    norm_flat = torch.stack([frame_transform(img) for img in flat], dim=0)
    videos = norm_flat.view(batch_size, num_frames, 3, 224, 224).to(device)

    captions = [
        "A person pouring water into a glass.",
        "A dog running across a green lawn.",
        "Someone typing on a laptop at a desk.",
        "A chef chopping vegetables in a kitchen.",
        "Cars driving along a city street at night.",
        "A cyclist riding on a mountain trail.",
        "A cat jumping onto a window sill.",
        "Two people playing tennis on an outdoor court.",
    ][:batch_size]

    encoded = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return (
        videos,
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


@torch.no_grad()
def compute_recall_at_k(video_emb: torch.Tensor, text_emb: torch.Tensor, ks=(1, 5)):
    """
    计算配对集上的双向 Recall@K：
    - T2V: 给定文本 j，检索匹配视频 j
    - V2T: 给定视频 i，检索匹配文本 i
    """
    video_emb = F.normalize(video_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    sim_v2t = torch.matmul(video_emb, text_emb.t())  # (N, N)
    sim_t2v = sim_v2t.t()  # (N, N)

    num_items = video_emb.size(0)
    ground_truth = torch.arange(num_items, device=video_emb.device).unsqueeze(1)

    metrics = {}
    for k in ks:
        top_k = min(k, num_items)

        t2v_top = sim_t2v.topk(top_k, dim=1).indices
        t2v_hits = (t2v_top == ground_truth).any(dim=1).float().mean().item() * 100.0

        v2t_top = sim_v2t.topk(top_k, dim=1).indices
        v2t_hits = (v2t_top == ground_truth).any(dim=1).float().mean().item() * 100.0

        metrics[f"T2V_R@{k}"] = t2v_hits
        metrics[f"V2T_R@{k}"] = v2t_hits

    return metrics


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 512
    batch_size = 8
    num_frames = 8
    epochs = 3
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    frame_transform = build_frame_transform()

    video_encoder = VideoEncoder(embedding_dim=embedding_dim, max_frames=32).to(device)
    text_encoder = TextEncoder(model_name=model_name, embedding_dim=embedding_dim).to(device)
    criterion = SymmetricInfoNCELoss(init_temperature=0.07).to(device)

    optimizer = optim.AdamW(
        [
            {"params": video_encoder.parameters(), "lr": 1e-4},
            {"params": text_encoder.parameters(), "lr": 2e-5},
            {"params": [criterion.logit_scale], "lr": 1e-3},
        ],
        weight_decay=1e-2,
    )

    videos, input_ids, attention_mask = create_dummy_batch(
        tokenizer=tokenizer,
        frame_transform=frame_transform,
        batch_size=batch_size,
        num_frames=num_frames,
        device=device,
    )

    for epoch in range(1, epochs + 1):
        video_encoder.train()
        text_encoder.train()
        criterion.train()

        optimizer.zero_grad()
        video_emb = video_encoder(videos)
        text_emb = text_encoder(input_ids, attention_mask)
        loss = criterion(video_emb, text_emb)
        loss.backward()
        optimizer.step()

        video_encoder.eval()
        text_encoder.eval()
        with torch.no_grad():
            eval_v = video_encoder(videos)
            eval_t = text_encoder(input_ids, attention_mask)
            recalls = compute_recall_at_k(eval_v, eval_t, ks=(1, 5))

        print(
            f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | "
            f"T2V R@1: {recalls['T2V_R@1']:.1f}% | V2T R@1: {recalls['V2T_R@1']:.1f}%"
        )


if __name__ == "__main__":
    train_and_evaluate()