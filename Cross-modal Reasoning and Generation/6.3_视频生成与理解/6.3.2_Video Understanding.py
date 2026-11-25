import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
import numpy as np


class ComplexVideoDataset(Dataset):
    """
    视频帧数据集：
    期望每条样本 video_frames 形状为 (T, H, W, C)，数值范围为 [0, 255] 的 uint8 或 np.ndarray/Tensor。
    """

    def __init__(self, video_frames, labels, processor: CLIPProcessor):
        self.video_frames = video_frames
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frames = self.video_frames[idx]
        label = int(self.labels[idx])

        if isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = np.asarray(frames)

        if np.issubdtype(frames_np.dtype, np.floating):
            if frames_np.min() >= 0.0 and frames_np.max() <= 1.0:
                frames_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                normed = 1.0 / (1.0 + np.exp(-np.clip(frames_np, -10.0, 10.0)))
                frames_np = (normed * 255.0).astype(np.uint8)
        elif frames_np.dtype != np.uint8:
            frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

        processed = self.processor(images=list(frames_np), return_tensors="pt")
        pixel_values = processed["pixel_values"]  # (T, 3, 224, 224)

        return pixel_values, torch.tensor(label, dtype=torch.long)


class ComplexVideoUnderstandingModel(nn.Module):
    """
    基于 CLIP 帧级视觉编码 + 时间位置编码 + TransformerEncoder 的视频分类模型。
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 10,
        max_frames: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        clip_embed_dim = self.clip_model.config.projection_dim  # ViT-B/32 默认为 512

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.input_proj = (
            nn.Identity()
            if clip_embed_dim == d_model
            else nn.Linear(clip_embed_dim, d_model)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _extract_clip_image_features(self, flat_frames: torch.Tensor) -> torch.Tensor:
        """
        兼容 transformers 4.x（直接返回 Tensor）与 5.x（返回 BaseModelOutputWithPooling 等输出对象）。
        """
        can_grad = self.clip_model.training and any(
            p.requires_grad for p in self.clip_model.parameters()
        )
        with torch.set_grad_enabled(can_grad):
            raw = self.clip_model.get_image_features(pixel_values=flat_frames)

            if isinstance(raw, torch.Tensor):
                feats = raw
            elif hasattr(raw, "image_embeds") and raw.image_embeds is not None:
                feats = raw.image_embeds
            elif hasattr(raw, "pooler_output") and raw.pooler_output is not None:
                feats = raw.pooler_output
                # 若 pooler_output 未经过 visual_projection 且维度不等于 projection_dim，则显式投影
                if (
                    feats.size(-1) != self.clip_model.config.projection_dim
                    and hasattr(self.clip_model, "visual_projection")
                ):
                    feats = self.clip_model.visual_projection(feats)
            elif hasattr(raw, "last_hidden_state") and raw.last_hidden_state is not None:
                feats = raw.last_hidden_state[:, 0, :]
                if (
                    feats.size(-1) != self.clip_model.config.projection_dim
                    and hasattr(self.clip_model, "visual_projection")
                ):
                    feats = self.clip_model.visual_projection(feats)
            else:
                raise TypeError(f"Unsupported output type from get_image_features: {type(raw)}")

            return F.normalize(feats, p=2, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, T, C, H, W)
        返回: (B, num_classes)
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        if num_frames > self.pos_embedding.size(1):
            raise ValueError(
                f"num_frames ({num_frames}) exceeds max_frames ({self.pos_embedding.size(1)})."
            )

        flat_frames = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        image_features = self._extract_clip_image_features(flat_frames)

        seq_features = self.input_proj(image_features).view(batch_size, num_frames, -1)
        seq_features = seq_features + self.pos_embedding[:, :num_frames, :]
        seq_features = self.pos_dropout(seq_features)

        temporal_out = self.temporal_encoder(seq_features)

        pooled = self.norm(temporal_out.mean(dim=1))
        logits = self.classifier(pooled)
        return logits


def train_model(model, dataloader, optimizer, criterion, epochs=5, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for frames, labels in dataloader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    for frames, labels in dataloader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(frames)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")
    return acc


if __name__ == "__main__":
    num_samples = 100
    num_frames = 16
    num_classes = 10

    rng = np.random.default_rng(42)
    video_frames = [
        rng.integers(0, 256, size=(num_frames, 64, 64, 3), dtype=np.uint8)
        for _ in range(num_samples)
    ]
    labels = rng.integers(0, num_classes, size=num_samples)

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    dataset = ComplexVideoDataset(video_frames, labels, processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexVideoUnderstandingModel(
        model_name=model_id,
        num_classes=num_classes,
        max_frames=32,
        freeze_clip=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)

    train_model(model, dataloader, optimizer, criterion, epochs=5, device=device)
    evaluate_model(model, dataloader, device=device)