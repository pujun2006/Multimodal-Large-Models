import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModalityEncoder(nn.Module):
    """单模态基础特征编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DisentangledBranch(nn.Module):
    """将单模态特征拆分为共享表征（Shared）与模态特有表征（Specific）"""

    def __init__(self, in_dim: int, branch_dim: int):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, branch_dim),
            nn.ReLU(inplace=True),
        )
        self.specific_proj = nn.Sequential(
            nn.Linear(in_dim, branch_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_proj(feat)      # [B, branch_dim]
        specific = self.specific_proj(feat)  # [B, branch_dim]
        return shared, specific


class SharedFusionLayer(nn.Module):
    """跨模态共享特征融合层"""

    def __init__(self, branch_dim: int, out_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(branch_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, img_shared: torch.Tensor, text_shared: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((img_shared, text_shared), dim=1))


class TaskHead(nn.Module):
    """任务专属预测头"""

    def __init__(self, in_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def compute_orthogonal_loss(shared: torch.Tensor, specific: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算共享表征与特有表征之间的正交解耦损失：
    先对样本维度做 L2 归一化，计算跨子空间 Gram 矩阵并取 Frobenius 范数平方，
    鼓励 shared 与 specific 在表征空间中趋于正交、降低冗余。
    """
    s_norm = F.normalize(shared, p=2, dim=-1, eps=eps)      # [B, D]
    p_norm = F.normalize(specific, p=2, dim=-1, eps=eps)  # [B, D]

    # Gram 矩阵: [D, D]，衡量共享维度与特有维度之间的相关性
    cross_gram = torch.matmul(s_norm.transpose(0, 1), p_norm) / shared.size(0)
    return torch.sum(cross_gram ** 2)


class DisentangledMultiTaskModel(nn.Module):
    """带有共享/特有双分支的多模态多任务模型"""

    def __init__(
        self,
        img_input_dim: int,
        text_input_dim: int,
        hidden_dim: int,
        branch_dim: int,
        img_output_dim: int,
        text_output_dim: int,
    ):
        super().__init__()
        self.image_encoder = ModalityEncoder(img_input_dim, hidden_dim)
        self.text_encoder = ModalityEncoder(text_input_dim, hidden_dim)

        self.img_branch = DisentangledBranch(hidden_dim, branch_dim)
        self.text_branch = DisentangledBranch(hidden_dim, branch_dim)

        self.shared_fusion = SharedFusionLayer(branch_dim, hidden_dim)

        # 各任务头融合“跨模态共享表征”与“对应模态特有表征”
        self.img_task_head = TaskHead(hidden_dim + branch_dim, img_output_dim)
        self.text_task_head = TaskHead(hidden_dim + branch_dim, text_output_dim)

    def forward(self, img_input: torch.Tensor, text_input: torch.Tensor):
        img_base = self.image_encoder(img_input)
        text_base = self.text_encoder(text_input)

        # 1. 拆分为共享与特有双分支
        img_shared, img_specific = self.img_branch(img_base)
        text_shared, text_specific = self.text_branch(text_base)

        # 2. 共享表征跨模态融合
        joint_shared = self.shared_fusion(img_shared, text_shared)

        # 3. 结合对应模态特有表征进行任务预测
        img_task_feat = torch.cat((joint_shared, img_specific), dim=1)
        text_task_feat = torch.cat((joint_shared, text_specific), dim=1)

        img_output = self.img_task_head(img_task_feat)
        text_output = self.text_task_head(text_task_feat)

        aux_feats = {
            "img_shared": img_shared,
            "img_specific": img_specific,
            "text_shared": text_shared,
            "text_specific": text_specific,
        }
        return img_output, text_output, aux_feats


def main():
    torch.manual_seed(42)

    img_input_dim = 256
    text_input_dim = 128
    hidden_dim = 64
    branch_dim = 32
    img_output_dim = 10  # 图像 10 分类
    text_output_dim = 2   # 文本 2 分类
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DisentangledMultiTaskModel(
        img_input_dim=img_input_dim,
        text_input_dim=text_input_dim,
        hidden_dim=hidden_dim,
        branch_dim=branch_dim,
        img_output_dim=img_output_dim,
        text_output_dim=text_output_dim,
    ).to(device)

    img_criterion = nn.CrossEntropyLoss()
    text_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 模拟输入与标签
    img_data = torch.randn(batch_size, img_input_dim, device=device)
    text_data = torch.randn(batch_size, text_input_dim, device=device)
    img_labels = torch.randint(0, img_output_dim, (batch_size,), device=device)
    text_labels = torch.randint(0, text_output_dim, (batch_size,), device=device)

    # 损失权重配置
    w_img, w_text = 1.0, 1.0
    w_ortho = 0.1  # 正交解耦损失权重

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        img_outputs, text_outputs, aux = model(img_data, text_data)

        img_loss = img_criterion(img_outputs, img_labels)
        text_loss = text_criterion(text_outputs, text_labels)

        # 分别计算图像与文本“共享 vs. 特有”正交解耦损失
        ortho_img = compute_orthogonal_loss(aux["img_shared"], aux["img_specific"])
        ortho_text = compute_orthogonal_loss(aux["text_shared"], aux["text_specific"])
        ortho_loss = ortho_img + ortho_text

        total_loss = w_img * img_loss + w_text * text_loss + w_ortho * ortho_loss

        total_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1} | Img Loss: {img_loss.item():.4f} | "
            f"Text Loss: {text_loss.item():.4f} | Ortho Loss: {ortho_loss.item():.4f} | "
            f"Total Loss: {total_loss.item():.4f}"
        )

    # 模型评估
    model.eval()
    test_img_data = torch.randn(batch_size, img_input_dim, device=device)
    test_text_data = torch.randn(batch_size, text_input_dim, device=device)

    with torch.no_grad():
        test_img_outputs, test_text_outputs, _ = model(test_img_data, test_text_data)
        img_predictions = torch.argmax(test_img_outputs, dim=1)
        text_predictions = torch.argmax(test_text_outputs, dim=1)

    print("\nImage Predictions:", img_predictions)
    print("Text Predictions:", text_predictions)


if __name__ == "__main__":
    main()