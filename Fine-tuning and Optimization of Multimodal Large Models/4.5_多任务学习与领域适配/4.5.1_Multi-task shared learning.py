import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, ViTModel


# 1. 多模态多任务数据集（支持单模态缺失：has_image=False 时用零张量+mask屏蔽）
class MultimodalMultiTaskDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=64, img_size=224):
        """
        samples: list of dict
          - task: "itm" (图文是否匹配 0/1) 或 "mm_sentiment" (图文情感 0/1/2)
          - text: str
          - image: tensor [3, H, W] 或 None
          - has_image: bool
          - label: int
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoded = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        has_img = bool(item.get("has_image", item.get("image") is not None))
        if has_img and item.get("image") is not None:
            pixel_values = item["image"]
        else:
            pixel_values = torch.zeros(3, self.img_size, self.img_size)
            has_img = False

        return {
            "task": item["task"],
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "has_image": torch.tensor(has_img, dtype=torch.bool),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


# 2. 多模态多任务共享模型
class MultimodalMultiTaskModel(nn.Module):
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        vision_model_name="google/vit-base-patch16-224",
        proj_dim=512,
        num_sentiment_classes=3,
    ):
        super().__init__()
        # 双模态编码器（实际工程中也可替换为统一多模态大模型/视觉语言底座）
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)

        text_dim = self.text_encoder.config.hidden_size
        vision_dim = self.vision_encoder.config.hidden_size

        # 模态投影层：将不同模态特征对齐到同一维度
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.vision_proj = nn.Linear(vision_dim, proj_dim)
        self.missing_img_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)

        # 共享跨模态交互层：文本 Query 关注图像 Patch Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(proj_dim * 2)
        self.shared_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 多任务专属输出头
        self.itm_head = nn.Linear(proj_dim, 2)  # 任务1：图文匹配二分类
        self.sentiment_head = nn.Linear(proj_dim, num_sentiment_classes)  # 任务2：图文多分类

        # 可选：任务对数方差参数（用于自动平衡多任务损失尺度）
        self.log_vars = nn.ParameterDict({
            "itm": nn.Parameter(torch.zeros(1)),
            "mm_sentiment": nn.Parameter(torch.zeros(1)),
        })

    def encode_multimodal(self, input_ids, attention_mask, pixel_values, has_image):
        # 文本序列特征 [B, L, D_t] -> [B, L, P]
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_pooled = text_seq[:, 0, :]  # [CLS] 投影向量 [B, P]

        # 图像 Patch 特征 [B, N, D_v] -> [B, N, P]
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        vis_seq = self.vision_proj(vis_out.last_hidden_state)

        # 若样本缺失图像，用可学习占位向量替换，避免噪声图像特征污染融合
        batch_size, num_patches, _ = vis_seq.shape
        fallback_vis = self.missing_img_token.expand(batch_size, num_patches, -1)
        vis_seq = torch.where(has_image.view(batch_size, 1, 1), vis_seq, fallback_vis)

        # 跨模态注意力融合：让文本表征对齐相关视觉区域
        attn_out, _ = self.cross_attn(
            query=text_seq,
            key=vis_seq,
            value=vis_seq,
        )
        vis_pooled = attn_out.mean(dim=1)

        # 拼接并经过共享融合网络，得到多任务共享表征 [B, P]
        fused = self.fusion_norm(torch.cat([text_pooled, vis_pooled], dim=-1))
        shared_repr = self.shared_mlp(fused)

        # 同时返回归一化后的全局特征，便于做跨模态对比对齐辅助损失
        text_emb = F.normalize(text_pooled, dim=-1)
        vis_emb = F.normalize(vis_pooled, dim=-1)
        return shared_repr, text_emb, vis_emb

    def forward(self, task, input_ids, attention_mask, pixel_values, has_image):
        shared_repr, text_emb, vis_emb = self.encode_multimodal(
            input_ids, attention_mask, pixel_values, has_image
        )
        if task == "itm":
            logits = self.itm_head(shared_repr)
        elif task == "mm_sentiment":
            logits = self.sentiment_head(shared_repr)
        else:
            raise ValueError(f"Unknown task: {task}")
        return logits, text_emb, vis_emb


def compute_multitask_loss(model, batch, ce_loss_fn, align_weight=0.1):
    task = batch["task"][0]  # 每个混合 batch 采样单一任务或按任务分组计算
    logits, text_emb, vis_emb = model(
        task=task,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        has_image=batch["has_image"],
    )
    task_loss = ce_loss_fn(logits, batch["label"])

    # 辅助对齐损失：仅在 batch 内存在有效图像时计算，促进图文空间对齐
    valid_mask = batch["has_image"]
    if valid_mask.sum() >= 2:
        t_valid = text_emb[valid_mask]
        v_valid = vis_emb[valid_mask]
        # 简单的余弦对齐损失（正对拉近）
        align_loss = (1.0 - (t_valid * v_valid).sum(dim=-1)).mean()
    else:
        align_loss = torch.tensor(0.0, device=logits.device)

    # 不确定性加权（Kendall et al. 风格）：自动调节不同任务的梯度权重
    log_var = model.log_vars[task]
    weighted_task_loss = torch.exp(-log_var) * task_loss + 0.5 * log_var
    total_loss = weighted_task_loss + align_weight * align_loss
    return total_loss, task_loss.item(), align_loss.item()


def train_multimodal_multitask(model, loader_itm, loader_sent, epochs=3, device="cpu"):
    model.to(device)
    ce_loss_fn = nn.CrossEntropyLoss()

    # 分层学习率：预训练编码器用较小 LR，投影/融合/任务头用稍大 LR
    encoder_params = list(model.text_encoder.parameters()) + list(model.vision_encoder.parameters())
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("text_encoder.") and not n.startswith("vision_encoder.")
    ]
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-4, "weight_decay": 0.01},
    ])

    for epoch in range(epochs):
        model.train()
        iters = {
            "itm": iter(loader_itm),
            "mm_sentiment": iter(loader_sent),
        }
        # 按任务样本比例构建 step 序列（避免整轮硬切分先训完任务A再训任务B）
        schedule = ["itm"] * len(loader_itm) + ["mm_sentiment"] * len(loader_sent)
        random.shuffle(schedule)

        stats = {"itm": [], "mm_sentiment": [], "align": []}
        for task_name in schedule:
            batch = next(iters[task_name])
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss, t_loss, a_loss = compute_multitask_loss(model, batch, ce_loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            stats[task_name].append(t_loss)
            stats["align"].append(a_loss)

        itm_avg = sum(stats["itm"]) / max(len(stats["itm"]), 1)
        sent_avg = sum(stats["mm_sentiment"]) / max(len(stats["mm_sentiment"]), 1)
        align_avg = sum(stats["align"]) / max(len(stats["align"]), 1)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"ITM Loss: {itm_avg:.4f} | "
            f"MM-Sentiment Loss: {sent_avg:.4f} | "
            f"Align Loss: {align_avg:.4f}"
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 构造演示样本（实际训练请替换为真实图像张量与业务标注）
    itm_samples = [
        {"task": "itm", "text": "A golden retriever running on the grass", "image": torch.randn(3, 224, 224), "has_image": True, "label": 1},
        {"task": "itm", "text": "A city skyline at night", "image": torch.randn(3, 224, 224), "has_image": True, "label": 0},
        {"task": "itm", "text": "Fresh fruit bowl on a table", "image": torch.randn(3, 224, 224), "has_image": True, "label": 1},
        {"task": "itm", "text": "Snowy mountain peak", "image": torch.randn(3, 224, 224), "has_image": True, "label": 0},
    ]
    sent_samples = [
        {"task": "mm_sentiment", "text": "Love this new camera setup!", "image": torch.randn(3, 224, 224), "has_image": True, "label": 2},
        {"task": "mm_sentiment", "text": "Packaging arrived damaged.", "image": torch.randn(3, 224, 224), "has_image": True, "label": 0},
        {"task": "mm_sentiment", "text": "Standard delivery status update.", "image": None, "has_image": False, "label": 1},  # 缺图样本
        {"task": "mm_sentiment", "text": "Great battery life so far.", "image": torch.randn(3, 224, 224), "has_image": True, "label": 2},
    ]

    ds_itm = MultimodalMultiTaskDataset(itm_samples, tokenizer)
    ds_sent = MultimodalMultiTaskDataset(sent_samples, tokenizer)

    loader_itm = DataLoader(ds_itm, batch_size=2, shuffle=True)
    loader_sent = DataLoader(ds_sent, batch_size=2, shuffle=True)

    model = MultimodalMultiTaskModel()
    train_multimodal_multitask(model, loader_itm, loader_sent, epochs=2, device=device)
    torch.save(model.state_dict(), "multimodal_multitask_model.pth")
