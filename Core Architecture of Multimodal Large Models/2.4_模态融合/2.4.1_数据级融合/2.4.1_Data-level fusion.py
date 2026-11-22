import torch
import torch.nn as nn
import torch.optim as optim


class DataLevelFusionModel(nn.Module):
    """
    数据级融合（早期融合）模型：
    在输入阶段直接将 RGB 图像 [B, 3, H, W] 与深度图 [B, 1, H, W]
    沿通道维度拼接为 [B, 4, H, W]，再送入统一骨干网络进行联合特征提取与分类。
    """

    def __init__(self, hidden_dim: int = 128, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        # 第一层卷积直接接收 4 通道输入（3 通道 RGB + 1 通道 Depth）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        rgb_mask: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
            rgb_mask: 可选 [B, 1, 1, 1]，1 表示保留，0 表示该模态缺失/置零
            depth_mask: 可选 [B, 1, 1, 1]，1 表示保留，0 表示该模态缺失/置零
        """
        if rgb_mask is not None:
            rgb = rgb * rgb_mask
        if depth_mask is not None:
            depth = depth * depth_mask

        # 核心步骤：在原始/低阶输入阶段直接按通道维度拼接 [B, 4, H, W]
        fused_input = torch.cat([rgb, depth], dim=1)

        feats = self.stem(fused_input)
        logits = self.classifier(feats)
        return logits


def main():
    torch.manual_seed(42)

    batch_size = 16
    img_h, img_w = 128, 128
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DataLevelFusionModel(hidden_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 模拟已空间对齐的 RGB 与 Depth 输入数据
    rgb_data = torch.randn(batch_size, 3, img_h, img_w, device=device)
    depth_data = torch.randn(batch_size, 1, img_h, img_w, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # 模拟训练时随机丢弃部分样本的深度通道（增强模态缺失鲁棒性）
    depth_mask = (torch.rand(batch_size, 1, 1, 1, device=device) > 0.2).float()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(rgb_data, depth_data, depth_mask=depth_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # 推理阶段评估
    model.eval()
    test_rgb = torch.randn(batch_size, 3, img_h, img_w, device=device)
    test_depth = torch.randn(batch_size, 1, img_h, img_w, device=device)

    with torch.no_grad():
        test_outputs = model(test_rgb, test_depth)
        predictions = torch.argmax(test_outputs, dim=1)

    print("\nTest Predictions:", predictions)


if __name__ == "__main__":
    main()