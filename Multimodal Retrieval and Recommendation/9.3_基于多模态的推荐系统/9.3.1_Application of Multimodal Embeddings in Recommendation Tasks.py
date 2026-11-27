import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_synthetic_multimodal_data(
    num_users: int = 200,
    num_items: int = 300,
    user_feat_dim: int = 16,
    image_feat_dim: int = 32,
    text_feat_dim: int = 32,
    latent_dim: int = 16,
    rating_min: float = 0.0,
    rating_max: float = 5.0,
    noise_std: float = 0.25,
):
    """
    生成带有潜在跨模态偏好结构的模拟多模态推荐数据。
    真实场景中可替换为预训练图像/文本编码器提取的特征与真实交互记录。
    """
    user_features = torch.randn(num_users, user_feat_dim)
    item_image_features = torch.randn(num_items, image_feat_dim)
    item_text_features = torch.randn(num_items, text_feat_dim)

    w_u = torch.randn(user_feat_dim, latent_dim) / np.sqrt(user_feat_dim)
    w_img = torch.randn(image_feat_dim, latent_dim) / np.sqrt(image_feat_dim)
    w_txt = torch.randn(text_feat_dim, latent_dim) / np.sqrt(text_feat_dim)

    u_latent = torch.tanh(user_features @ w_u)
    img_latent = torch.tanh(item_image_features @ w_img)
    txt_latent = torch.tanh(item_text_features @ w_txt)
    item_latent = 0.5 * img_latent + 0.5 * txt_latent

    raw_affinity = u_latent @ item_latent.T
    scaled_affinity = torch.sigmoid(raw_affinity) * (rating_max - rating_min) + rating_min
    noise = torch.randn_like(scaled_affinity) * noise_std
    ratings_matrix = torch.clamp(scaled_affinity + noise, rating_min, rating_max)

    return user_features, item_image_features, item_text_features, ratings_matrix


def build_interaction_dataset(
    user_features: torch.Tensor,
    item_image_features: torch.Tensor,
    item_text_features: torch.Tensor,
    ratings_matrix: torch.Tensor,
    train_ratio: float = 0.8,
    batch_size: int = 64,
):
    """将 (user_idx, item_idx, rating) 打包为训练与测试 DataLoader。"""
    num_users, num_items = ratings_matrix.shape
    pairs = [(u, i) for u in range(num_users) for i in range(num_items)]
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def to_tensor_dataset(pair_list):
        u_idx = torch.tensor([p[0] for p in pair_list], dtype=torch.long)
        i_idx = torch.tensor([p[1] for p in pair_list], dtype=torch.long)
        u_feat = user_features[u_idx]
        img_feat = item_image_features[i_idx]
        txt_feat = item_text_features[i_idx]
        ratings = ratings_matrix[u_idx, i_idx]
        return TensorDataset(u_feat, img_feat, txt_feat, ratings)

    train_loader = DataLoader(to_tensor_dataset(train_pairs), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(to_tensor_dataset(test_pairs), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class ModalityEncoder(nn.Module):
    """通用特征编码器，将特定模态特征映射到统一嵌入空间。"""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalRecommender(nn.Module):
    """
    多模态推荐模型：
    1. 分别编码用户特征、物品图像特征、物品文本特征到统一嵌入空间；
    2. 使用模态注意力权重动态融合图像与文本嵌入，得到物品多模态表征；
    3. 拼接用户嵌入与物品嵌入，通过多层感知机预测评分并映射到 [rating_min, rating_max]。
    """

    def __init__(
        self,
        user_feat_dim: int,
        image_feat_dim: int,
        text_feat_dim: int,
        embed_dim: int = 32,
        rating_min: float = 0.0,
        rating_max: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rating_min = rating_min
        self.rating_max = rating_max

        self.user_encoder = ModalityEncoder(user_feat_dim, 64, embed_dim, dropout)
        self.image_encoder = ModalityEncoder(image_feat_dim, 64, embed_dim, dropout)
        self.text_encoder = ModalityEncoder(text_feat_dim, 64, embed_dim, dropout)

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
        )

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_item(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        img_embed = self.image_encoder(img_feat)
        txt_embed = self.text_encoder(txt_feat)

        gate_logits = self.fusion_gate(torch.cat([img_embed, txt_embed], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)

        w_img = weights[:, 0:1]
        w_txt = weights[:, 1:2]
        return w_img * img_embed + w_txt * txt_embed

    def forward(
        self,
        user_feat: torch.Tensor,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
    ) -> torch.Tensor:
        user_embed = self.user_encoder(user_feat)
        item_embed = self.encode_item(img_feat, txt_feat)

        interaction = torch.cat([user_embed, item_embed], dim=-1)
        raw_score = self.scorer(interaction).squeeze(-1)
        return torch.sigmoid(raw_score) * (self.rating_max - self.rating_min) + self.rating_min


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """在完整数据集上评估 MSE 与 RMSE。"""
    model.eval()
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for u_feat, img_feat, txt_feat, ratings in data_loader:
            u_feat = u_feat.to(device)
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            ratings = ratings.to(device)

            preds = model(u_feat, img_feat, txt_feat)
            batch_loss = criterion(preds, ratings).item()
            batch_size = ratings.size(0)
            total_squared_error += batch_loss * batch_size
            total_samples += batch_size

    mse = total_squared_error / max(total_samples, 1)
    rmse = float(np.sqrt(mse))
    return mse, rmse


def predict_single(
    model: nn.Module,
    user_idx: int,
    item_idx: int,
    user_features: torch.Tensor,
    item_image_features: torch.Tensor,
    item_text_features: torch.Tensor,
    device: torch.device,
) -> float:
    """对单个 (user_idx, item_idx) 进行评分推断。"""
    model.eval()
    with torch.no_grad():
        u_feat = user_features[user_idx].unsqueeze(0).to(device)
        img_feat = item_image_features[item_idx].unsqueeze(0).to(device)
        txt_feat = item_text_features[item_idx].unsqueeze(0).to(device)
        pred = model(u_feat, img_feat, txt_feat)
    return float(pred.item())


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rating_min, rating_max = 0.0, 5.0
    epochs = 20
    batch_size = 64
    learning_rate = 1e-3

    user_features, item_image_features, item_text_features, ratings_matrix = (
        generate_synthetic_multimodal_data(
            num_users=200,
            num_items=300,
            user_feat_dim=16,
            image_feat_dim=32,
            text_feat_dim=32,
            latent_dim=16,
            rating_min=rating_min,
            rating_max=rating_max,
        )
    )

    train_loader, test_loader = build_interaction_dataset(
        user_features=user_features,
        item_image_features=item_image_features,
        item_text_features=item_text_features,
        ratings_matrix=ratings_matrix,
        train_ratio=0.8,
        batch_size=batch_size,
    )

    model = MultiModalRecommender(
        user_feat_dim=user_features.size(1),
        image_feat_dim=item_image_features.size(1),
        text_feat_dim=item_text_features.size(1),
        embed_dim=32,
        rating_min=rating_min,
        rating_max=rating_max,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
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

            batch_size_cur = ratings.size(0)
            running_loss += loss.item() * batch_size_cur
            total_samples += batch_size_cur

        train_mse = running_loss / max(total_samples, 1)
        if epoch % 5 == 0 or epoch == 1:
            val_mse, val_rmse = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train MSE: {train_mse:.4f} | Test MSE: {val_mse:.4f} | Test RMSE: {val_rmse:.4f}"
            )

    test_mse, test_rmse = evaluate(model, test_loader, criterion, device)
    print("\n=== 最终测试集评估 ===")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    model.eval()
    sample_u, sample_img, sample_txt, sample_r = next(iter(test_loader))
    with torch.no_grad():
        sample_preds = model(sample_u.to(device), sample_img.to(device), sample_txt.to(device)).cpu()

    print("\n预测评分 vs. 真实评分（前 5 条）:")
    for idx in range(min(5, sample_preds.size(0))):
        print(f"  样本 {idx}: 预测 = {sample_preds[idx].item():.3f} | 真实 = {sample_r[idx].item():.3f}")

    demo_pred = predict_single(
        model=model,
        user_idx=0,
        item_idx=0,
        user_features=user_features,
        item_image_features=item_image_features,
        item_text_features=item_text_features,
        device=device,
    )
    print(f"\n用户 0 对物品 0 的预测评分: {demo_pred:.3f}（真实评分: {ratings_matrix[0, 0].item():.3f}）")


if __name__ == "__main__":
    main()