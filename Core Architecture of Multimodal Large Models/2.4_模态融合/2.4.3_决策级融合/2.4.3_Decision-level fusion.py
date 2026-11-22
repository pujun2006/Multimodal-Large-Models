import torch
import torch.nn as nn
import torch.optim as optim


class UnimodalBranch(nn.Module):
    """单模态独立分支：完成各自的特征提取与独立预测输出"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出该模态独立的预测 logits: [B, num_classes]
        return self.net(x)


class DecisionLevelFusionModel(nn.Module):
    """
    决策级融合（晚期融合）模型：
    1. 文本分支与图像分支各自独立给出预测结果（probabilities）。
    2. 在决策层按可学习权重对各分支预测概率进行加权融合（支持样本级模态可用性掩码）。
    """

    def __init__(
        self,
        text_input_dim: int,
        image_input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.text_branch = UnimodalBranch(text_input_dim, hidden_dim, num_classes, dropout_rate)
        self.image_branch = UnimodalBranch(image_input_dim, hidden_dim, num_classes, dropout_rate)

        # 决策层可学习分支权重参数（经 softmax 转为正权重）
        self.branch_weight_logits = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        text_input: torch.Tensor,
        image_input: torch.Tensor,
        text_avail: torch.Tensor | None = None,
        image_avail: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            text_input: [B, text_input_dim]
            image_input: [B, image_input_dim]
            text_avail: 可选 [B, 1]，1 表示文本分支有效，0 表示缺失/失效
            image_avail: 可选 [B, 1]，1 表示图像分支有效，0 表示缺失/失效
        Returns:
            fused_log_probs: [B, num_classes]，可直接传入 nn.NLLLoss()
            aux_outputs: 包含各单模态 logits、概率与实际融合权重的字典
        """
        batch_size = text_input.size(0)
        device = text_input.device

        # 1. 各模态独立完成推理与预测
        text_logits = self.text_branch(text_input)    # [B, C]
        image_logits = self.image_branch(image_input)  # [B, C]

        text_probs = torch.softmax(text_logits, dim=-1)    # [B, C]
        image_probs = torch.softmax(image_logits, dim=-1)  # [B, C]

        # 2. 获取基础分支权重并结合样本级可用性掩码做归一化
        base_weights = torch.softmax(self.branch_weight_logits, dim=0)  # [2]
        w_text = base_weights[0].expand(batch_size, 1)
        w_image = base_weights[1].expand(batch_size, 1)

        if text_avail is None:
            text_avail = torch.ones(batch_size, 1, device=device)
        if image_avail is None:
            image_avail = torch.ones(batch_size, 1, device=device)

        eff_w_text = w_text * text_avail
        eff_w_image = w_image * image_avail
        weight_sum = (eff_w_text + eff_w_image).clamp(min=1e-8)

        norm_w_text = eff_w_text / weight_sum
        norm_w_image = eff_w_image / weight_sum

        # 3. 决策层概率加权融合
        fused_probs = norm_w_text * text_probs + norm_w_image * image_probs
        fused_log_probs = torch.log(fused_probs.clamp(min=1e-8))

        aux_outputs = {
            "text_logits": text_logits,
            "image_logits": image_logits,
            "text_probs": text_probs,
            "image_probs": image_probs,
            "norm_w_text": norm_w_text,
            "norm_w_image": norm_w_image,
        }
        return fused_log_probs, aux_outputs


def main():
    torch.manual_seed(42)

    batch_size = 16
    text_input_dim = 128
    image_input_dim = 256
    hidden_dim = 64
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionLevelFusionModel(
        text_input_dim=text_input_dim,
        image_input_dim=image_input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    # 融合输出为对数概率，使用 NLLLoss 计算分类损失
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    text_data = torch.randn(batch_size, text_input_dim, device=device)
    image_data = torch.randn(batch_size, image_input_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # 模拟部分样本图像分支缺失（0 表示该样本图像不可用）
    image_avail = (torch.rand(batch_size, 1, device=device) > 0.25).float()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        fused_log_probs, aux = model(text_data, image_data, image_avail=image_avail)
        loss = criterion(fused_log_probs, labels)
        loss.backward()
        optimizer.step()

        learned_w = torch.softmax(model.branch_weight_logits, dim=0).detach().cpu().numpy()
        print(
            f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | "
            f"Base Weights (Text, Image): ({learned_w[0]:.3f}, {learned_w[1]:.3f})"
        )

    # 推理评估：演示某条测试样本图像缺失时的自动权重回退
    model.eval()
    test_text = torch.randn(4, text_input_dim, device=device)
    test_image = torch.randn(4, image_input_dim, device=device)
    test_image_avail = torch.tensor([[1.0], [1.0], [0.0], [1.0]], device=device)

    with torch.no_grad():
        test_log_probs, test_aux = model(test_text, test_image, image_avail=test_image_avail)
        predictions = torch.argmax(test_log_probs, dim=-1)

    print("\nTest Predictions:", predictions)
    print("Effective Text Weights per sample:", test_aux["norm_w_text"].squeeze(-1))
    print("Effective Image Weights per sample:", test_aux["norm_w_image"].squeeze(-1))


if __name__ == "__main__":
    main()