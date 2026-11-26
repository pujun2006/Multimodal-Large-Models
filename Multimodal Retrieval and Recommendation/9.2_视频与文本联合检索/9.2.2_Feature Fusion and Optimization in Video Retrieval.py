import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ==========================================
# 1. 模拟数据集构建（含潜在语义簇 + 训练/测试划分）
# ==========================================
def build_synthetic_dataset(
    num_samples: int = 120,
    video_dim: int = 512,
    text_dim: int = 300,
    latent_dim: int = 128,
    noise_std: float = 0.25,
):
    """
    构造带潜在语义结构的配对视频-文本特征：
    每个样本 i 有一个共享语义原型 z_i，视频特征与文本特征分别由不同投影矩阵 + 噪声生成，
    使模型能够真正学习跨模态对应关系而非记忆纯随机噪声。
    """
    latent_concepts = F.normalize(torch.randn(num_samples, latent_dim), p=2, dim=1)

    # 固定随机投影矩阵模拟不同模态特征提取器
    w_v = torch.randn(latent_dim, video_dim) / np.sqrt(latent_dim)
    w_t = torch.randn(latent_dim, text_dim) / np.sqrt(latent_dim)

    video_features = latent_concepts @ w_v + noise_std * torch.randn(num_samples, video_dim)
    text_features = latent_concepts @ w_t + noise_std * torch.randn(num_samples, text_dim)

    return video_features, text_features


# ==========================================
# 2. 编码器 + 跨模态特征融合模块
# ==========================================
class ModalityEncoder(nn.Module):
    """单模态特征投影编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalFusionBlock(nn.Module):
    """
    跨模态特征融合模块：
    将视频表征与文本表征视为长度为 1 的模态 token 序列，
    通过 MultiheadAttention 进行双向交互，再经残差前馈网络融合。
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, vid_embed: torch.Tensor, txt_embed: torch.Tensor):
        # vid_embed, txt_embed: [B, D] -> 拼接为 [B, 2, D]
        tokens = torch.stack([vid_embed, txt_embed], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        fused_vid = tokens[:, 0, :]
        fused_txt = tokens[:, 1, :]
        # 残差保留原始对齐向量，兼顾检索效率与融合表达
        return F.normalize(vid_embed + fused_vid, p=2, dim=-1), F.normalize(
            txt_embed + fused_txt, p=2, dim=-1
        )


class MultiModalVideoRetriever(nn.Module):
    def __init__(
        self,
        video_dim: int = 512,
        text_dim: int = 300,
        embed_dim: int = 256,
        num_heads: int = 4,
        init_temp: float = 0.07,
    ):
        super().__init__()
        self.video_encoder = ModalityEncoder(video_dim, 1024, embed_dim)
        self.text_encoder = ModalityEncoder(text_dim, 512, embed_dim)
        self.fusion = CrossModalFusionBlock(embed_dim=embed_dim, num_heads=num_heads)
        # 可学习的温度系数（对数参数化保证正值）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / init_temp))

    def encode_videos(self, video_x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.video_encoder(video_x), p=2, dim=-1)

    def encode_texts(self, text_x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(text_x), p=2, dim=-1)

    def forward(self, video_x: torch.Tensor, text_x: torch.Tensor):
        v_raw = self.encode_videos(video_x)
        t_raw = self.encode_texts(text_x)
        v_feat, t_feat = self.fusion(v_raw, t_raw)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return v_feat, t_feat, logit_scale


# ==========================================
# 3. 对称 InfoNCE 对比损失
# ==========================================
class SymmetricInfoNCELoss(nn.Module):
    """
    批内全负例对称对比损失：
    第 i 个文本对应第 i 个视频为正例，其余 N-1 个为负例；
    同时计算 Text->Video 与 Video->Text 双向交叉熵。
    """

    def forward(
        self, vid_embed: torch.Tensor, txt_embed: torch.Tensor, logit_scale: torch.Tensor
    ) -> torch.Tensor:
        # logits_per_text[i, j] 表示第 i 条文本与第 j 条视频的缩放余弦相似度
        logits_per_text = logit_scale * (txt_embed @ vid_embed.t())
        logits_per_video = logits_per_text.t()

        batch_size = vid_embed.size(0)
        targets = torch.arange(batch_size, device=vid_embed.device)

        loss_t2v = F.cross_entropy(logits_per_text, targets)
        loss_v2t = F.cross_entropy(logits_per_video, targets)
        return 0.5 * (loss_t2v + loss_v2t)


# ==========================================
# 4. 检索评估指标（T2V Recall@K & MRR）
# ==========================================
@torch.no_grad()
def evaluate_t2v(model: MultiModalVideoRetriever, test_videos: torch.Tensor, test_texts: torch.Tensor, device: torch.device):
    model.eval()
    v_emb = model.encode_videos(test_videos.to(device))
    t_emb = model.encode_texts(test_texts.to(device))

    # sim_matrix[i, j]: 第 i 条查询文本对第 j 个视频的相似度，形状 [N_test, N_test]
    sim_matrix = t_emb @ v_emb.t()
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    num_queries = test_texts.size(0)
    ranks = []
    for i in range(num_queries):
        # 真实匹配的视频索引即为 i
        matched_pos = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item()
        ranks.append(matched_pos + 1)  # 1-based 排名

    ranks_arr = np.array(ranks)
    r1 = np.mean(ranks_arr <= 1) * 100.0
    r5 = np.mean(ranks_arr <= min(5, num_queries)) * 100.0
    mrr = np.mean(1.0 / ranks_arr) * 100.0
    return {"R@1": r1, "R@5": r5, "MRR": mrr}


# ==========================================
# 5. 训练与推理主流程
# ==========================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_total = 120
    train_size = 90
    video_dim = 512
    text_dim = 300
    embed_dim = 256
    batch_size = 30
    epochs = 25

    all_videos, all_texts = build_synthetic_dataset(
        num_samples=num_total, video_dim=video_dim, text_dim=text_dim
    )
    train_videos, test_videos = all_videos[:train_size], all_videos[train_size:]
    train_texts, test_texts = all_texts[:train_size], all_texts[train_size:]

    model = MultiModalVideoRetriever(
        video_dim=video_dim, text_dim=text_dim, embed_dim=embed_dim, num_heads=4
    ).to(device)
    criterion = SymmetricInfoNCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"使用设备: {device} | 训练集: {train_size} | 测试集: {num_total - train_size}")
    init_metrics = evaluate_t2v(model, test_videos, test_texts, device)
    print(
        f"训练前测试集 -> R@1: {init_metrics['R@1']:.2f}% | "
        f"R@5: {init_metrics['R@5']:.2f}% | MRR: {init_metrics['MRR']:.2f}%"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_size)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, train_size, batch_size):
            idx = perm[start : start + batch_size]
            v_batch = train_videos[idx].to(device)
            t_batch = train_texts[idx].to(device)

            optimizer.zero_grad()
            v_emb, t_emb, logit_scale = model(v_batch, t_batch)
            loss = criterion(v_emb, t_emb, logit_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_t2v(model, test_videos, test_texts, device)
            print(
                f"Epoch [{epoch:02d}/{epochs}] | Loss: {epoch_loss / steps:.4f} | "
                f"Test R@1: {metrics['R@1']:.2f}% | R@5: {metrics['R@5']:.2f}% | MRR: {metrics['MRR']:.2f}%"
            )

    # 推理示例：用测试集第 0 条文本检索 Top-3 视频
    model.eval()
    with torch.no_grad():
        query_t = model.encode_texts(test_texts[0:1].to(device))
        cand_v = model.encode_videos(test_videos.to(device))
        scores = (query_t @ cand_v.t()).squeeze(0)
        top_vals, top_idxs = torch.topk(scores, k=3)

    print("\n=== 测试集单条检索示例（目标正确视频索引为 0）===")
    for rank, (vid_idx, score) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), 1):
        print(f"  Top-{rank}: 测试视频索引={vid_idx}, 余弦相似度={score:.4f}")


if __name__ == "__main__":
    main()