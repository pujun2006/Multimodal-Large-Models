import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast


class JointVisualSemanticNavModel(nn.Module):
    """
    视觉与语义联合优化导航模型：
    1. 视觉分支提取空间特征网格 [B, N_vis, D] + 可学习空间位置编码
    2. 语义分支提取指令 token 表征 [B, L_txt, D]
    3. 跨模态交叉注意力实现视觉区域与语义指代对齐
    4. 联合损失包含：任务生成损失 + 跨模态对比对齐损失 + 语义保真损失
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        align_weight: float = 0.25,
        sem_reg_weight: float = 0.05,
        temperature: float = 0.07,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.align_weight = align_weight
        self.sem_reg_weight = sem_reg_weight
        self.temperature = temperature

        # 1. 视觉骨干（输出空间特征图 [B, 2048, 7, 7] -> 49 个空间 token）
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.vision_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, 49, hidden_dim) * 0.02)

        # 2. 语义编码器
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 3. 跨模态联合交互层（文本查询视觉区域 + 前馈融合）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 4. 自回归语言生成器（与 GPT2TokenizerFast 严格对齐）
        self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
        lm_dim = self.lm_model.config.n_embd
        self.joint_to_lm = nn.Sequential(
            nn.Linear(hidden_dim, lm_dim),
            nn.LayerNorm(lm_dim),
        )

        if freeze_backbones:
            for p in self.vision_backbone.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """返回 [B, 49, hidden_dim] 的空间视觉 token 序列"""
        feat_map = self.vision_backbone(images)  # [B, 2048, 7, 7]
        vis_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, 49, 2048]
        vis_tokens = self.vision_proj(vis_tokens) + self.spatial_pos_emb
        return vis_tokens

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """返回 [B, L_txt, hidden_dim] 的语义 token 序列"""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(outputs.last_hidden_state)

    def fuse_joint_features(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        融合视觉与语义特征：
        返回 joint_tokens [B, L_txt, D] 和 attn_weights [B, L_txt, 49]
        """
        attended_txt, attn_weights = self.cross_attn(
            query=txt_tokens,
            key=vis_tokens,
            value=vis_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm1(txt_tokens + attended_txt)
        joint_tokens = self.norm2(x + self.ffn(x))
        return joint_tokens, attn_weights

    def compute_joint_losses(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        joint_tokens: torch.Tensor,
        lm_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合优化目标：
        - total_loss: 总损失
        - lm_loss: 导航序列生成损失
        - align_loss: 跨模态对称对比损失（InfoNCE）
        - sem_reg_loss: 语义保真正则，防止联合表征过度偏离原始语义
        """
        batch_size = vis_tokens.size(0)

        # 跨模态对比对齐损失（batch 内样本 >= 2 时计算 InfoNCE，否则退化为余弦对齐）
        vis_vec = F.normalize(vis_tokens.mean(dim=1), dim=-1)
        jnt_vec = F.normalize(joint_tokens.mean(dim=1), dim=-1)

        if batch_size >= 2:
            logits = (vis_vec @ jnt_vec.t()) / self.temperature
            targets = torch.arange(batch_size, device=vis_tokens.device)
            align_loss = 0.5 * (
                F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
            )
        else:
            align_loss = 1.0 - F.cosine_similarity(vis_vec, jnt_vec, dim=-1).mean()

        # 语义保真约束
        sem_reg_loss = F.mse_loss(joint_tokens, txt_tokens)

        total_loss = (
            lm_loss
            + self.align_weight * align_loss
            + self.sem_reg_weight * sem_reg_loss
        )
        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "align_loss": align_loss,
            "sem_reg_loss": sem_reg_loss,
        }

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
    ):
        vis_tokens = self.encode_vision(images)
        txt_tokens = self.encode_text(text_input_ids, text_attention_mask)
        joint_tokens, attn_weights = self.fuse_joint_features(
            vis_tokens, txt_tokens, text_attention_mask
        )

        # 将联合特征映射为 LM 前缀提示 [B, L_txt, n_embd]
        prefix_embeds = self.joint_to_lm(joint_tokens)

        if target_ids is not None:
            tgt_embeds = self.lm_model.transformer.wte(target_ids)
            inputs_embeds = torch.cat([prefix_embeds, tgt_embeds], dim=1)

            # 前缀 token 不参与自回归交叉熵计算（label 设为 -100）
            prefix_labels = torch.full(
                (target_ids.size(0), prefix_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=target_ids.device,
            )
            if target_attention_mask is not None:
                masked_targets = target_ids.masked_fill(target_attention_mask == 0, -100)
            else:
                masked_targets = target_ids

            full_labels = torch.cat([prefix_labels, masked_targets], dim=1)
            lm_outputs = self.lm_model(inputs_embeds=inputs_embeds, labels=full_labels)
            losses = self.compute_joint_losses(
                vis_tokens=vis_tokens,
                txt_tokens=txt_tokens,
                joint_tokens=joint_tokens,
                lm_loss=lm_outputs.loss,
            )
            losses["attn_weights"] = attn_weights
            return losses

        return {
            "prefix_embeds": prefix_embeds,
            "attn_weights": attn_weights,
        }

    @torch.no_grad()
    def generate_navigation_plan(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        tokenizer: GPT2TokenizerFast,
        max_new_tokens: int = 32,
    ) -> List[str]:
        self.eval()
        out = self.forward(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            target_ids=None,
        )
        prefix_embeds = out["prefix_embeds"]
        prefix_mask = torch.ones(
            prefix_embeds.size()[:2], dtype=torch.long, device=prefix_embeds.device
        )

        gen_ids = self.lm_model.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return [
            tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in gen_ids
        ]


class NavigationDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        bert_tokenizer: BertTokenizerFast,
        gpt_tokenizer: GPT2TokenizerFast,
        max_text_len: int = 64,
        max_target_len: int = 64,
    ):
        self.samples = samples
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.max_text_len = max_text_len
        self.max_target_len = max_target_len
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        with Image.open(item["image_path"]) as img:
            image_tensor = self.image_transform(img.convert("RGB"))

        text_enc = self.bert_tokenizer(
            item["instruction"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        target_enc = self.gpt_tokenizer(
            item["target_plan"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "text_input_ids": text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "target_ids": target_enc["input_ids"].squeeze(0),
            "target_attention_mask": target_enc["attention_mask"].squeeze(0),
        }


def create_sample_images(output_dir: str = "sample_nav_maps") -> List[Dict[str, str]]:
    """先生成示例图像，再供 Dataset 读取，避免文件不存在错误"""
    os.makedirs(output_dir, exist_ok=True)
    img1_path = os.path.join(output_dir, "scene_1.png")
    img2_path = os.path.join(output_dir, "scene_2.png")

    img1 = Image.new("RGB", (224, 224), "white")
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([20, 20, 90, 90], outline="black", width=2)
    d1.rectangle([130, 130, 205, 205], outline="black", width=2)
    d1.text((30, 45), "Entry", fill="black")
    d1.text((140, 155), "Kitchen", fill="black")
    img1.save(img1_path)

    img2 = Image.new("RGB", (224, 224), "white")
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([25, 120, 95, 195], outline="black", width=2)
    d2.rectangle([125, 25, 200, 95], outline="black", width=2)
    d2.text((35, 145), "Hall", fill="black")
    d2.text((135, 50), "Study", fill="black")
    img2.save(img2_path)

    return [
        {
            "image_path": img1_path,
            "instruction": "Navigate from Entry to Kitchen passing through the central hall.",
            "target_plan": "Proceed forward from Entry, pass the central hall, and turn right into Kitchen.",
        },
        {
            "image_path": img2_path,
            "instruction": "Go from Hall to Study without crossing the blocked zone.",
            "target_plan": "Walk along the right corridor from Hall and enter Study on the upper right.",
        },
    ]


def train_model(
    model: JointVisualSemanticNavModel,
    dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-4,
    device: str = "cpu",
):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        metrics_sum = {"total_loss": 0.0, "lm_loss": 0.0, "align_loss": 0.0, "sem_reg_loss": 0.0}

        for batch in dataloader:
            images = batch["image"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)

            optimizer.zero_grad()
            losses = model(
                images=images,
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
                target_ids=target_ids,
                target_attention_mask=target_mask,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in metrics_sum:
                metrics_sum[k] += losses[k].item()

        scheduler.step()
        n_batches = len(dataloader)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Total: {metrics_sum['total_loss']/n_batches:.4f} | "
            f"LM: {metrics_sum['lm_loss']/n_batches:.4f} | "
            f"Align: {metrics_sum['align_loss']/n_batches:.4f} | "
            f"SemReg: {metrics_sum['sem_reg_loss']/n_batches:.4f}"
        )


def evaluate_samples(
    model: JointVisualSemanticNavModel,
    samples: List[Dict[str, str]],
    bert_tokenizer: BertTokenizerFast,
    gpt_tokenizer: GPT2TokenizerFast,
    device: str = "cpu",
):
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    for item in samples:
        with Image.open(item["image_path"]) as img:
            img_tensor = image_transform(img.convert("RGB")).unsqueeze(0).to(device)

        enc = bert_tokenizer(
            item["instruction"],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        preds = model.generate_navigation_plan(
            images=img_tensor,
            text_input_ids=enc["input_ids"].to(device),
            text_attention_mask=enc["attention_mask"].to(device),
            tokenizer=gpt_tokenizer,
            max_new_tokens=28,
        )
        print("-" * 60)
        print("输入指令：", item["instruction"])
        print("目标参考：", item["target_plan"])
        print("生成规划：", preds[0])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = create_sample_images()

    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    dataset = NavigationDataset(samples, bert_tokenizer, gpt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = JointVisualSemanticNavModel(
        hidden_dim=768,
        num_heads=8,
        align_weight=0.25,
        sem_reg_weight=0.05,
        freeze_backbones=False,
    ).to(device)

    train_model(model, dataloader, epochs=2, lr=1e-4, device=device)
    evaluate_samples(model, samples, bert_tokenizer, gpt_tokenizer, device=device)