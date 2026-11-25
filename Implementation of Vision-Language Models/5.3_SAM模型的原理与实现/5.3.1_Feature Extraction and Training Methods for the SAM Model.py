import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==========================================
# 1. 二维正弦位置编码（SAM 点/框坐标编码基础）
# ==========================================
class PositionEmbeddingRandom(nn.Module):
    """使用随机高斯频率矩阵生成二维坐标位置编码"""

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [..., 2]，归一化到 [0, 1] 后映射到 [-1, 1]
        coords = 2.0 * coords - 1.0
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2.0 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(self, coords: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        h, w = image_size
        norm_coords = torch.empty_like(coords, dtype=torch.float32)
        norm_coords[..., 0] = coords[..., 0] / float(w)
        norm_coords[..., 1] = coords[..., 1] / float(h)
        return self._pe_encoding(norm_coords)

    def forward_grid(self, grid_size: tuple[int, int]) -> torch.Tensor:
        h, w = grid_size
        y = (torch.arange(h, device=self.positional_encoding_gaussian_matrix.device) + 0.5) / h
        x = (torch.arange(w, device=self.positional_encoding_gaussian_matrix.device) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W, 2]
        pe = self._pe_encoding(coords)  # [H*W, C]
        return pe.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]


# ==========================================
# 2. 图像编码器（特征提取 + 通道对齐）
# ==========================================
class ImageEncoder(nn.Module):
    """
    提取图像空间特征图并投影到统一嵌入维度 embed_dim。
    真实 SAM 使用 MAE 预训练 ViT（如 1024x1024 -> 64x64x256）；
    此处用卷积主干下采样（例如 256x256 -> 32x32x256），保证形状严格对齐且可离线运行。
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.neck(feat)  # [B, embed_dim, H_feat, W_feat]


# ==========================================
# 3. 提示编码器（点提示 + 框提示）
# ==========================================
class PromptEncoder(nn.Module):
    """
    将稀疏提示（点坐标 + 前景/背景标签，以及可选边界框）编码为 Token 序列。
    """

    def __init__(self, embed_dim: int = 256, image_size: tuple[int, int] = (256, 256)):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)

        # 点类别嵌入：0=背景点，1=前景点
        self.point_embeddings = nn.Embedding(2, embed_dim)
        # 边界框角点嵌入：0=左上角，1=右下角
        self.box_corner_embeddings = nn.Embedding(2, embed_dim)
        # 空提示占位 Token（当无显式提示或填充点时使用）
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        points: (coords [B, N_pts, 2], labels [B, N_pts] 其中 1=前景, 0=背景, -1=无效填充)
        boxes:  [B, 4] 格式为 (x1, y1, x2, y2)
        返回: sparse_prompt_tokens [B, N_tokens, embed_dim]
        """
        tokens_list = []

        if points is not None:
            coords, labels = points
            pt_pe = self.pe_layer.forward_with_coords(coords + 0.5, self.image_size)
            valid_mask = (labels >= 0).unsqueeze(-1)
            safe_labels = labels.clamp(min=0, max=1)
            type_emb = self.point_embeddings(safe_labels)
            pt_tokens = torch.where(
                valid_mask,
                pt_pe + type_emb,
                self.not_a_point_embed.weight.unsqueeze(0).expand_as(pt_pe),
            )
            tokens_list.append(pt_tokens)

        if boxes is not None:
            corners = boxes.view(-1, 2, 2) + 0.5
            box_pe = self.pe_layer.forward_with_coords(corners, self.image_size)
            corner_ids = torch.tensor([0, 1], device=boxes.device).unsqueeze(0).expand(boxes.size(0), -1)
            box_tokens = box_pe + self.box_corner_embeddings(corner_ids)
            tokens_list.append(box_tokens)

        if not tokens_list:
            b = 1 if points is None else points[0].size(0)
            return self.not_a_point_embed.weight.unsqueeze(0).expand(b, 1, self.embed_dim)

        return torch.cat(tokens_list, dim=1)


# ==========================================
# 4. 轻量级掩码解码器（多掩码输出 + IoU 预测头）
# ==========================================
class LightweightMaskDecoder(nn.Module):
    """
    通过交叉注意力让提示 Token 与图像特征交互，
    生成 num_mask_tokens 个候选掩码及对应的 IoU 预测分数。
    """

    def __init__(self, embed_dim: int = 256, num_multimask_outputs: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs

        # 1 个 IoU 预测 Token + num_multimask_outputs 个掩码输出 Token
        num_output_tokens = 1 + num_multimask_outputs
        self.output_tokens = nn.Embedding(num_output_tokens, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 上采样空间特征并动态生成掩码
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.mask_hyper_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_multimask_outputs),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_tokens: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          masks:    [B, num_multimask_outputs, H, W]
          iou_pred: [B, num_multimask_outputs]
        """
        b, _, _, _ = image_embeddings.shape
        out_tokens = self.output_tokens.weight.unsqueeze(0).expand(b, -1, -1)  # [B, 1+M, C]
        queries = torch.cat([out_tokens, sparse_prompt_tokens], dim=1)  # [B, 1+M+N_pts, C]

        img_tokens = (image_embeddings + image_pe).flatten(2).transpose(1, 2)  # [B, H*W, C]
        attended_queries = self.decoder(tgt=queries, memory=img_tokens)

        iou_token_out = attended_queries[:, 0, :]  # [B, C]
        mask_tokens_out = attended_queries[:, 1 : 1 + self.num_multimask_outputs, :]  # [B, M, C]

        up_feat = self.upscale(image_embeddings)  # [B, C//2, 2H, 2W]
        hyper_in = self.mask_hyper_mlp(mask_tokens_out)  # [B, M, C//2]
        masks = torch.einsum("bmc,bchw->bmhw", hyper_in, up_feat)  # [B, M, 2H, 2W]
        masks = F.interpolate(masks, size=target_size, mode="bilinear", align_corners=False)

        iou_pred = self.iou_head(iou_token_out)  # [B, M]
        return masks, iou_pred


# ==========================================
# 5. 完整 MiniSAM 模型封装
# ==========================================
class MiniSAM(nn.Module):
    def __init__(self, embed_dim: int = 256, image_size: tuple[int, int] = (256, 256)):
        super().__init__()
        self.image_size = image_size
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim, image_size=image_size)
        self.mask_decoder = LightweightMaskDecoder(embed_dim=embed_dim, num_multimask_outputs=3)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """图像特征只需计算一次，可重复用于多次交互提示推理"""
        img_emb = self.image_encoder(images)
        _, _, h, w = img_emb.shape
        img_pe = self.pe_layer.forward_grid((h, w)).to(img_emb.device)
        return img_emb, img_pe

    def forward(
        self,
        images: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_emb, img_pe = self.encode_image(images)
        sparse_tokens = self.prompt_encoder(points=points, boxes=boxes)
        masks, iou_pred = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=img_pe,
            sparse_prompt_tokens=sparse_tokens,
            target_size=self.image_size,
        )
        return masks, iou_pred


# ==========================================
# 6. 损失函数与 IoU 计算
# ==========================================
def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    loss = alpha_t * ((1.0 - pt) ** gamma) * bce
    return loss.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


@torch.no_grad()
def compute_batch_iou(pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """计算预测掩码与真值掩码的真实 IoU，用于监督 IoU 预测头"""
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()
    inter = (pred_bin * gt_bin).sum(dim=(-2, -1))
    union = (pred_bin + gt_bin).clamp(max=1.0).sum(dim=(-2, -1)).clamp(min=1.0)
    return inter / union  # [B, M]


# ==========================================
# 7. 交互式模拟采样函数（初始点 + 误差纠偏点）
# ==========================================
@torch.no_grad()
def sample_initial_point_from_gt(gt_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从真值前景区域随机采样 1 个初始正样本点 (x, y)"""
    b, _, h, w = gt_mask.shape
    coords = torch.zeros((b, 1, 2), dtype=torch.float32, device=gt_mask.device)
    labels = torch.ones((b, 1), dtype=torch.long, device=gt_mask.device)

    for i in range(b):
        ys, xs = torch.where(gt_mask[i, 0] > 0.5)
        if len(ys) > 0:
            idx = torch.randint(0, len(ys), (1,)).item()
            coords[i, 0, 0] = xs[idx].float()
            coords[i, 0, 1] = ys[idx].float()
        else:
            coords[i, 0, 0] = w / 2.0
            coords[i, 0, 1] = h / 2.0
    return coords, labels


@torch.no_grad()
def sample_correction_point(
    pred_logits: torch.Tensor, gt_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对比当前预测掩码与 GT：
    - 若 GT 为前景但预测为背景（漏检 FN），采样前景点 (label=1)
    - 若 GT 为背景但预测为前景（误检 FP），采样背景点 (label=0)
    """
    b, _, h, w = gt_mask.shape
    pred_bin = torch.sigmoid(pred_logits) > 0.5
    gt_bin = gt_mask[:, 0] > 0.5

    new_coords = torch.zeros((b, 1, 2), dtype=torch.float32, device=gt_mask.device)
    new_labels = torch.ones((b, 1), dtype=torch.long, device=gt_mask.device)

    for i in range(b):
        fn_ys, fn_xs = torch.where(gt_bin[i] & (~pred_bin[i]))
        fp_ys, fp_xs = torch.where((~gt_bin[i]) & pred_bin[i])

        if len(fn_ys) >= len(fp_ys) and len(fn_ys) > 0:
            idx = torch.randint(0, len(fn_ys), (1,)).item()
            new_coords[i, 0, 0] = fn_xs[idx].float()
            new_coords[i, 0, 1] = fn_ys[idx].float()
            new_labels[i, 0] = 1
        elif len(fp_ys) > 0:
            idx = torch.randint(0, len(fp_ys), (1,)).item()
            new_coords[i, 0, 0] = fp_xs[idx].float()
            new_coords[i, 0, 1] = fp_ys[idx].float()
            new_labels[i, 0] = 0
        else:
            ys, xs = torch.where(gt_bin[i])
            if len(ys) > 0:
                idx = torch.randint(0, len(ys), (1,)).item()
                new_coords[i, 0, 0] = xs[idx].float()
                new_coords[i, 0, 1] = ys[idx].float()
            else:
                new_coords[i, 0, 0] = w / 2.0
                new_coords[i, 0, 1] = h / 2.0
            new_labels[i, 0] = 1

    return new_coords, new_labels


# ==========================================
# 8. 合成数据构造 + 训练与推理演示
# ==========================================
def make_synthetic_batch(
    batch_size: int,
    img_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[float, float]]]:
    """生成带有明显前景/背景对比的合成图像、目标掩码及圆心坐标"""
    h, w = img_size
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    images_list = []
    masks_list = []
    centers = []

    for i in range(batch_size):
        cx = 128.0 + (i - (batch_size - 1) / 2.0) * 16.0
        cy = 128.0
        radius = 42.0 + i * 4.0

        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2).float()
        bg = 0.15 + 0.05 * torch.randn(3, h, w, device=device)
        fg = 0.85 + 0.05 * torch.randn(3, h, w, device=device)
        img = torch.where(mask.unsqueeze(0) > 0.5, fg, bg).clamp(0.0, 1.0)

        images_list.append(img)
        masks_list.append(mask.unsqueeze(0))
        centers.append((cx, cy))

    images = torch.stack(images_list, dim=0)
    gt_masks = torch.stack(masks_list, dim=0)
    return images, gt_masks, centers


def train_and_infer_demo():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (256, 256)
    batch_size = 4
    epochs = 40

    model = MiniSAM(embed_dim=256, image_size=img_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    images, gt_masks, centers = make_synthetic_batch(
        batch_size=batch_size, img_size=img_size, device=device
    )

    model.train()
    num_interactive_steps = 3

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        coords, labels = sample_initial_point_from_gt(gt_masks)

        total_loss = 0.0
        for _ in range(num_interactive_steps):
            masks_pred, iou_pred = model(images, points=(coords, labels))
            gt_expanded = gt_masks.expand_as(masks_pred)

            loss_focal = torch.stack(
                [focal_loss(masks_pred[:, m], gt_masks[:, 0]) for m in range(masks_pred.size(1))]
            ).mean()
            loss_dice = torch.stack(
                [dice_loss(masks_pred[:, m], gt_masks[:, 0]) for m in range(masks_pred.size(1))]
            ).mean()

            gt_iou = compute_batch_iou(masks_pred, gt_expanded)
            loss_iou = F.mse_loss(torch.sigmoid(iou_pred), gt_iou)

            step_loss = 20.0 * loss_focal + loss_dice + loss_iou
            total_loss = total_loss + step_loss

            best_m = torch.argmax(iou_pred.mean(dim=0))
            next_coord, next_label = sample_correction_point(masks_pred[:, best_m], gt_masks)
            coords = torch.cat([coords, next_coord], dim=1)
            labels = torch.cat([labels, next_label], dim=1)

        total_loss = total_loss / num_interactive_steps
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                sample_iou = compute_batch_iou(masks_pred, gt_expanded).mean().item()
            print(
                f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss.item():.4f} | "
                f"Batch IoU: {sample_iou:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    # 推理阶段：使用第 0 张图圆心附近的前景点提示进行验证
    model.eval()
    with torch.no_grad():
        test_img = images[0:1]
        test_gt = gt_masks[0, 0]
        cx0, cy0 = centers[0]
        test_point = torch.tensor([[[cx0, cy0]]], dtype=torch.float32, device=device)
        test_label = torch.ones((1, 1), dtype=torch.long, device=device)

        pred_masks, pred_ious = model(test_img, points=(test_point, test_label))
        pred_probs_all = torch.sigmoid(pred_masks[0])  # [3, H, W]
        iou_scores = torch.sigmoid(pred_ious[0])       # [3]

        best_idx = int(torch.argmax(iou_scores).item())
        best_prob = pred_probs_all[best_idx]
        binary_mask = (best_prob > 0.5).to(torch.uint8)

        pred_fg = int(binary_mask.sum().item())
        gt_fg = int(test_gt.sum().item())
        inter = int(((binary_mask > 0) & (test_gt > 0.5)).sum().item())
        union = int(((binary_mask > 0) | (test_gt > 0.5)).sum().item())
        real_iou = inter / max(union, 1)

        print(
            f"推理完成 | 候选 IoU 预测: {[round(x, 3) for x in iou_scores.tolist()]} | "
            f"选中索引: {best_idx} | 概率范围: [{best_prob.min().item():.3f}, {best_prob.max().item():.3f}]"
        )
        print(
            f"掩码形状: {tuple(binary_mask.shape)} | 预测前景像素: {pred_fg} | "
            f"真实前景像素: {gt_fg} | 实际 IoU: {real_iou:.3f}"
        )


if __name__ == "__main__":
    train_and_infer_demo()