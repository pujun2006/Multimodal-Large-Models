import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from transformers import AutoModel, AutoTokenizer


class SpatialImageEncoder(nn.Module):
    """提取 ResNet-50 空间网格特征 [B, 2048, H', W'] -> [B, N_img, 2048]，默认 7x7=49 个空间 Patch。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)
        # 保留到 layer4 结束（输出 [B, 2048, 7, 7]），去掉 avgpool 和 fc
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.out_dim = 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(images)  # [B, C, H', W']
        bsz, channels, h_dim, w_dim = feat_map.shape
        # [B, C, H', W'] -> [B, H'*W', C]
        return feat_map.flatten(2).transpose(1, 2).contiguous()


class MaskedMultimodalReconstructor(nn.Module):
    """
    基于掩码的多模态重建模型：
    - 图像空间网格随机掩码 -> 跨模态 Transformer 融合 -> 预测被掩码位置原始图像特征（MSE）
    - 文本词元随机 [MASK] -> 跨模态 Transformer 融合 -> 预测被掩码位置原始 Token ID（CrossEntropy）
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        d_model: int = 512,
        nhead: int = 8,
        num_fusion_layers: int = 2,
        img_mask_ratio: float = 0.5,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        self.img_mask_ratio = img_mask_ratio
        self.d_model = d_model

        # 1. 视觉与文本编码骨干
        self.image_encoder = SpatialImageEncoder(pretrained=pretrained_cnn)
        img_raw_dim = self.image_encoder.out_dim  # 2048

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        txt_raw_dim = self.text_encoder.config.hidden_size  # 768
        vocab_size = self.text_encoder.config.vocab_size

        # 2. 模态投影、模态类型嵌入与位置嵌入
        self.img_proj_in = nn.Linear(img_raw_dim, d_model)
        self.txt_proj_in = nn.Linear(txt_raw_dim, d_model)

        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.img_mask_token, std=0.02)

        # 模态标识（0: 图像, 1: 文本）与最大序列长度位置编码
        self.modality_embed = nn.Embedding(2, d_model)
        self.max_img_tokens = 64
        self.max_txt_tokens = 128
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.max_img_tokens, d_model))
        self.txt_pos_embed = nn.Parameter(torch.zeros(1, self.max_txt_tokens, d_model))
        nn.init.normal_(self.img_pos_embed, std=0.02)
        nn.init.normal_(self.txt_pos_embed, std=0.02)

        # 3. 跨模态融合编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_modal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_fusion_layers
        )

        # 4. 图像与文本重建头
        self.img_recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, img_raw_dim),
        )
        self.txt_recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )

    def mask_text_inputs(self, texts: list[str], device: torch.device, mlm_prob: float = 0.15):
        """对文本输入执行随机 Token 掩码，返回带掩码的张量、原始 label_ids 与掩码布尔矩阵。"""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_txt_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()

        # 排除特殊 token（如 [CLS], [SEP], [PAD]）
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self.tokenizer.all_special_ids:
            special_tokens_mask |= input_ids == tid

        rand_prob = torch.rand(input_ids.shape, device=device)
        mlm_mask = (rand_prob < mlm_prob) & (~special_tokens_mask) & (attention_mask == 1)

        # 兜底：若某条样本未抽中任何可掩码 token，随机挑 1 个非特殊有效 token
        for b in range(input_ids.size(0)):
            if not mlm_mask[b].any():
                candidates = torch.where((~special_tokens_mask[b]) & (attention_mask[b] == 1))[0]
                if len(candidates) > 0:
                    chosen = candidates[torch.randint(0, len(candidates), (1,), device=device)]
                    mlm_mask[b, chosen] = True

        masked_input_ids = input_ids.clone()
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is not None:
            masked_input_ids[mlm_mask] = mask_token_id

        return masked_input_ids, attention_mask, labels, mlm_mask

    def mask_image_patches(self, img_tokens: torch.Tensor):
        """
        对图像空间 Patch 做随机掩码：
        - img_tokens: [B, N_img, C_img]
        - 返回掩码后的特征、原始目标特征（可选做 patch 级归一化）与掩码布尔矩阵 [B, N_img]
        """
        bsz, num_patches, _ = img_tokens.shape
        num_masked = max(1, int(num_patches * self.img_mask_ratio))

        noise = torch.rand(bsz, num_patches, device=img_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        img_mask = torch.zeros((bsz, num_patches), dtype=torch.bool, device=img_tokens.device)
        masked_indices = ids_shuffle[:, :num_masked]
        img_mask.scatter_(1, masked_indices, True)

        # 对目标特征做归一化，减少不同空间位置幅值差异带来的训练偏移
        target_img_feat = F.layer_norm(img_tokens.detach(), (img_tokens.size(-1),))

        # 投影到 d_model 并将被掩码位置替换为可学习的 img_mask_token
        img_proj = self.img_proj_in(img_tokens)
        mask_expanded = img_mask.unsqueeze(-1).expand_as(img_proj)
        masked_img_proj = torch.where(
            mask_expanded,
            self.img_mask_token.expand(bsz, num_patches, self.d_model),
            img_proj,
        )
        return masked_img_proj, target_img_feat, img_mask

    def forward(self, images: torch.Tensor, texts: list[str], mlm_prob: float = 0.15):
        device = images.device

        # 1. 提取原始图像空间特征（作为重建目标）并执行空间掩码
        raw_img_tokens = self.image_encoder(images)  # [B, N_img, 2048]
        n_img = raw_img_tokens.size(1)
        masked_img_proj, target_img_feat, img_mask = self.mask_image_patches(raw_img_tokens)

        # 2. 构造文本掩码输入并编码（同时提取原始 token id 作为重建目标）
        masked_input_ids, txt_attn_mask, txt_labels, txt_mask = self.mask_text_inputs(
            texts, device=device, mlm_prob=mlm_prob
        )
        txt_outputs = self.text_encoder(input_ids=masked_input_ids, attention_mask=txt_attn_mask)
        masked_txt_proj = self.txt_proj_in(txt_outputs.last_hidden_state)  # [B, N_txt, d_model]
        n_txt = masked_txt_proj.size(1)

        # 3. 叠加位置编码与模态嵌入，拼接成统一多模态序列
        img_seq = (
            masked_img_proj
            + self.img_pos_embed[:, :n_img, :]
            + self.modality_embed(torch.zeros((1, n_img), dtype=torch.long, device=device))
        )
        txt_seq = (
            masked_txt_proj
            + self.txt_pos_embed[:, :n_txt, :]
            + self.modality_embed(torch.ones((1, n_txt), dtype=torch.long, device=device))
        )
        multimodal_seq = torch.cat([img_seq, txt_seq], dim=1)  # [B, N_img + N_txt, d_model]

        # 4. 构造 padding mask（True 表示需要被 Transformer 忽略的 padding 位置）
        img_pad_mask = torch.zeros((images.size(0), n_img), dtype=torch.bool, device=device)
        txt_pad_mask = txt_attn_mask == 0
        src_key_padding_mask = torch.cat([img_pad_mask, txt_pad_mask], dim=1)

        # 5. 跨模态编码器交互
        fused_seq = self.cross_modal_encoder(
            multimodal_seq, src_key_padding_mask=src_key_padding_mask
        )

        # 6. 拆分回图像与文本分支，并仅在“被掩码位置”计算重建损失
        fused_img = fused_seq[:, :n_img, :]
        fused_txt = fused_seq[:, n_img:, :]

        pred_img_feat = self.img_recon_head(fused_img)  # [B, N_img, 2048]
        per_patch_mse = F.mse_loss(pred_img_feat, target_img_feat, reduction="none").mean(dim=-1)
        loss_img = (per_patch_mse * img_mask.float()).sum() / img_mask.float().sum().clamp_min(1.0)

        pred_txt_logits = self.txt_recon_head(fused_txt)  # [B, N_txt, vocab_size]
        # 未掩码或 padding 位置设为 -100，CrossEntropyLoss 会自动忽略
        masked_txt_targets = torch.where(
            txt_mask, txt_labels, torch.full_like(txt_labels, -100)
        )
        loss_txt = F.cross_entropy(
            pred_txt_logits.reshape(-1, pred_txt_logits.size(-1)),
            masked_txt_targets.reshape(-1),
            ignore_index=-100,
        )

        total_loss = loss_img + loss_txt
        return {
            "loss": total_loss,
            "loss_img": loss_img.detach(),
            "loss_txt": loss_txt.detach(),
            "pred_img_feat": pred_img_feat.detach(),
            "pred_txt_logits": pred_txt_logits.detach(),
        }


def dummy_dataloader(batch_size: int = 8, num_batches: int = 5, device: torch.device = torch.device("cpu")):
    sample_texts = [
        "A red panda sitting on a mossy branch in the forest.",
        "A researcher training a multimodal neural network on GPUs.",
        "Freshly cooked braised pork ribs with glossy dark sauce.",
        "An urban skyline at dusk with illuminated office towers.",
    ]
    for _ in range(num_batches):
        images = torch.rand(batch_size, 3, 224, 224, device=device)
        texts = [sample_texts[i % len(sample_texts)] for i in range(batch_size)]
        yield images, texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MaskedMultimodalReconstructor(
        text_model_name="bert-base-uncased",
        d_model=512,
        nhead=8,
        num_fusion_layers=2,
        img_mask_ratio=0.5,
        pretrained_cnn=False,  # 离线或快速冒烟测试可设 False；正式训练建议设 True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        for step, (images, texts) in enumerate(dummy_dataloader(batch_size=8, num_batches=5, device=device)):
            optimizer.zero_grad()
            outputs = model(images, texts, mlm_prob=0.15)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step + 1}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Img MSE (masked): {outputs['loss_img'].item():.4f} | "
                    f"Txt CE (masked): {outputs['loss_txt'].item():.4f}"
                )

    print("基于掩码的多模态重建训练示例运行完成。")


if __name__ == "__main__":
    main()
