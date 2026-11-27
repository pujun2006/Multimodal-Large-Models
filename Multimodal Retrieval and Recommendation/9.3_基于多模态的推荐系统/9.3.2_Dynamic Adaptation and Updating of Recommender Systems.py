import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 2024) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UserEncoder(nn.Module):
    """用户编码器，将用户特征映射到公共嵌入空间。"""

    def __init__(self, user_feat_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_feat_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ImageEncoder(nn.Module):
    """图像编码器，将物品图像特征映射到公共嵌入空间。"""

    def __init__(self, image_feat_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TextEncoder(nn.Module):
    """文本编码器，将物品文本特征映射到公共嵌入空间。"""

    def __init__(self, text_feat_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_feat_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DynamicMultiModalRec(nn.Module):
    """
    支持动态增量训练的多模态推荐模型：
    1. 分别编码用户特征、物品图像特征、物品文本特征到统一嵌入空间；
    2. 通过门控权重动态融合图像与文本嵌入；
    3. 融合用户嵌入与物品多模态嵌入，输出 [rating_min, rating_max] 范围内的预测评分。
    """

    def __init__(
        self,
        user_feat_dim: int,
        image_feat_dim: int,
        text_feat_dim: int,
        embed_dim: int = 16,
        rating_min: float = 0.0,
        rating_max: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rating_min = rating_min
        self.rating_max = rating_max

        self.user_encoder = UserEncoder(user_feat_dim, embed_dim, dropout)
        self.image_encoder = ImageEncoder(image_feat_dim, embed_dim, dropout)
        self.text_encoder = TextEncoder(text_feat_dim, embed_dim, dropout)

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode_item(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        image_embed = self.image_encoder(img_feat)
        text_embed = self.text_encoder(txt_feat)

        gate_logits = self.fusion_gate(torch.cat([image_embed, text_embed], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        w_img = weights[:, 0:1]
        w_txt = weights[:, 1:2]
        return w_img * image_embed + w_txt * text_embed

    def forward(
        self,
        user_feat: torch.Tensor,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
    ) -> torch.Tensor:
        user_embed = self.user_encoder(user_feat)
        item_embed = self.encode_item(img_feat, txt_feat)

        # 同时拼接差值/交互项，增强匹配信号
        prod_embed = user_embed * item_embed
        concat_vec = torch.cat([user_embed, item_embed, prod_embed], dim=-1)
        raw_score = self.fc_score(concat_vec).squeeze(-1)
        return torch.sigmoid(raw_score) * (self.rating_max - self.rating_min) + self.rating_min


def build_drifting_phases(
    num_users: int = 80,
    initial_items: int = 60,
    new_items_per_phase: int = 20,
    user_feat_dim: int = 12,
    image_feat_dim: int = 16,
    text_feat_dim: int = 16,
    latent_dim: int = 12,
    phase_count: int = 3,
    rating_min: float = 0.0,
    rating_max: float = 5.0,
    train_ratio: float = 0.8,
):
    """
    生成带有“阶段性分布偏移 + 新物品动态扩充”的多阶段推荐数据。
    每个阶段不仅包含新产生的交互，还引入阶段偏好矩阵与物品池扩充。
    """
    user_features = torch.randn(num_users, user_feat_dim)
    item_image_features = torch.randn(initial_items, image_feat_dim)
    item_text_features = torch.randn(initial_items, text_feat_dim)

    w_u = torch.randn(user_feat_dim, latent_dim) / np.sqrt(user_feat_dim)
    w_img = torch.randn(image_feat_dim, latent_dim) / np.sqrt(image_feat_dim)
    w_txt = torch.randn(text_feat_dim, latent_dim) / np.sqrt(text_feat_dim)

    phase_datasets = []
    cur_item_count = initial_items

    for phase_idx in range(phase_count):
        # 阶段 1 之后动态扩充新物品特征，模拟物品池持续上新
        if phase_idx > 0 and new_items_per_phase > 0:
            extra_img = torch.randn(new_items_per_phase, image_feat_dim)
            extra_txt = torch.randn(new_items_per_phase, text_feat_dim)
            item_image_features = torch.cat([item_image_features, extra_img], dim=0)
            item_text_features = torch.cat([item_text_features, extra_txt], dim=0)
            cur_item_count += new_items_per_phase

        # 每个阶段引入不同的偏好偏移与模态权重倾向，模拟兴趣漂移与热点变化
        phase_shift = torch.randn(latent_dim) * 0.6
        img_weight = float(np.clip(0.35 + 0.2 * phase_idx, 0.2, 0.8))
        txt_weight = 1.0 - img_weight

        u_latent = torch.tanh(user_features @ w_u + phase_shift)
        img_latent = torch.tanh(item_image_features @ w_img)
        txt_latent = torch.tanh(item_text_features @ w_txt)
        item_latent = img_weight * img_latent + txt_weight * txt_latent

        raw_affinity = u_latent @ item_latent.T
        scaled_affinity = torch.sigmoid(raw_affinity) * (rating_max - rating_min) + rating_min
        noise = torch.randn_like(scaled_affinity) * 0.2
        ratings_matrix = torch.clamp(scaled_affinity + noise, rating_min, rating_max)

        # 为当前阶段采样部分交互对（优先包含最新物品，模拟近期活跃交互）
        pairs = []
        recent_start = max(0, cur_item_count - (new_items_per_phase + 20))
        for u in range(num_users):
            sampled_items = random.sample(range(cur_item_count), k=min(15, cur_item_count))
            recent_items = random.sample(
                range(recent_start, cur_item_count),
                k=min(5, cur_item_count - recent_start),
            )
            for i in set(sampled_items + recent_items):
                pairs.append((u, i, float(ratings_matrix[u, i].item())))

        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]

        phase_datasets.append(
            {
                "train_pairs": train_pairs,
                "test_pairs": test_pairs,
                "num_items_at_phase": cur_item_count,
                "img_weight": img_weight,
            }
        )

    return user_features, item_image_features, item_text_features, phase_datasets


def make_loader(
    pairs,
    user_features: torch.Tensor,
    item_image_features: torch.Tensor,
    item_text_features: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """将 (u_idx, i_idx, rating) 列表转换为 DataLoader。"""
    u_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    i_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    ratings = torch.tensor([p[2] for p in pairs], dtype=torch.float32)

    dataset = TensorDataset(
        user_features[u_idx],
        item_image_features[i_idx],
        item_text_features[i_idx],
        ratings,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_mse(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """计算给定 DataLoader 上的加权平均 MSE。"""
    model.eval()
    total_se = 0.0
    total_samples = 0

    with torch.no_grad():
        for u_feat, img_feat, txt_feat, ratings in loader:
            u_feat = u_feat.to(device)
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            ratings = ratings.to(device)

            preds = model(u_feat, img_feat, txt_feat)
            batch_loss = criterion(preds, ratings).item()
            bs = ratings.size(0)
            total_se += batch_loss * bs
            total_samples += bs

    return total_se / max(total_samples, 1)


def main() -> None:
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_users = 80
    initial_items = 60
    new_items_per_phase = 20
    user_feat_dim = 12
    image_feat_dim = 16
    text_feat_dim = 16
    embed_dim = 16
    rating_min = 0.0
    rating_max = 5.0
    phase_count = 3
    epochs_per_phase = 8
    batch_size = 32
    replay_ratio = 0.25  # 混入 25% 历史回放样本，缓解遗忘

    user_features, item_image_features, item_text_features, phase_datasets = build_drifting_phases(
        num_users=num_users,
        initial_items=initial_items,
        new_items_per_phase=new_items_per_phase,
        user_feat_dim=user_feat_dim,
        image_feat_dim=image_feat_dim,
        text_feat_dim=text_feat_dim,
        latent_dim=12,
        phase_count=phase_count,
        rating_min=rating_min,
        rating_max=rating_max,
        train_ratio=0.8,
    )

    model = DynamicMultiModalRec(
        user_feat_dim=user_feat_dim,
        image_feat_dim=image_feat_dim,
        text_feat_dim=text_feat_dim,
        embed_dim=embed_dim,
        rating_min=rating_min,
        rating_max=rating_max,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    history_train_pool = []
    phase_test_loaders = []

    for phase_idx, p_data in enumerate(phase_datasets):
        print(
            f"\n=== Phase {phase_idx + 1}/{phase_count} "
            f"(累计物品数: {p_data['num_items_at_phase']}, 图像侧权重倾向: {p_data['img_weight']:.2f}) ==="
        )

        # 构造当前阶段训练集（可混入部分历史样本做经验回放）
        cur_train = list(p_data["train_pairs"])
        if history_train_pool and replay_ratio > 0:
            replay_k = min(len(history_train_pool), int(len(cur_train) * replay_ratio))
            cur_train.extend(random.sample(history_train_pool, k=replay_k))

        train_loader = make_loader(
            cur_train, user_features, item_image_features, item_text_features, batch_size=batch_size, shuffle=True
        )
        test_loader = make_loader(
            p_data["test_pairs"],
            user_features,
            item_image_features,
            item_text_features,
            batch_size=batch_size,
            shuffle=False,
        )
        phase_test_loaders.append(test_loader)

        # 增量更新前，先观察模型在当前新阶段测试集上的“零样本/未适配”误差
        pre_adapt_mse = evaluate_mse(model, test_loader, criterion, device)
        print(f"  适配前当前阶段测试 MSE: {pre_adapt_mse:.4f}")

        for epoch in range(1, epochs_per_phase + 1):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for u_feat, img_feat, txt_feat, ratings in train_loader:
                u_feat = u_feat.to(device)
                img_feat = img_feat.to(device)
                txt_feat = txt_feat.to(device)
                ratings = ratings.to(device)

                optimizer.zero_grad()
                preds = model(u_feat, img_feat, txt_feat)
                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()

                bs = ratings.size(0)
                running_loss += loss.item() * bs
                total_samples += bs

            if epoch == epochs_per_phase or epoch == 1:
                train_mse = running_loss / max(total_samples, 1)
                cur_test_mse = evaluate_mse(model, test_loader, criterion, device)
                print(
                    f"  Epoch [{epoch:02d}/{epochs_per_phase:02d}] | "
                    f"Train MSE: {train_mse:.4f} | 当前阶段 Test MSE: {cur_test_mse:.4f}"
                )

        # 跨阶段回测：检查当前模型在所有已经历阶段测试集上的表现（观察适配与抗遗忘）
        hist_report = [
            f"P{hist_idx + 1}: {evaluate_mse(model, hist_loader, criterion, device):.4f}"
            for hist_idx, hist_loader in enumerate(phase_test_loaders)
        ]
        print(f"  各阶段测试集 MSE 回测 -> {' | '.join(hist_report)}")

        history_train_pool.extend(p_data["train_pairs"])

    print("\n=== 最终阶段示例预测 ===")
    last_test_pairs = phase_datasets[-1]["test_pairs"]
    model.eval()
    with torch.no_grad():
        for u_idx, i_idx, true_rating in last_test_pairs[:5]:
            pred_score = model(
                user_features[u_idx].unsqueeze(0).to(device),
                item_image_features[i_idx].unsqueeze(0).to(device),
                item_text_features[i_idx].unsqueeze(0).to(device),
            )
            print(
                f"  用户 {u_idx} 对物品 {i_idx} 的真实评分: {true_rating:.3f}, "
                f"预测评分: {pred_score.item():.3f}"
            )


if __name__ == "__main__":
    main()