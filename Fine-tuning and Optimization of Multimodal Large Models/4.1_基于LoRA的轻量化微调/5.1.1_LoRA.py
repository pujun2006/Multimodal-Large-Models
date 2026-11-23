import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LoRALinear(nn.Module):
    """将普通 nn.Linear 包装为带 LoRA 低秩旁路的线性层。"""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_linear)}")

        self.base_linear = base_linear
        # 冻结原始线性层权重与偏置
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        # PyTorch nn.Linear 的前向为 x @ W.T + b（形状为 [..., out_features]）
        # lora_A: [in_features, rank] 随机高斯初始化
        # lora_B: [rank, out_features] 零初始化，保证训练初期增量输出为 0
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_linear(x)
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """将 LoRA 权重增量合并进 base_linear.weight，实现零额外推理开销。"""
        if self.merged:
            return
        # base_linear.weight 形状为 [out_features, in_features]
        # (x @ lora_A @ lora_B) * scaling 等价于 x @ (scaling * lora_A @ lora_B)
        # 对应权重矩阵增量 delta_W = scaling * (lora_A @ lora_B).T = scaling * (lora_B.T @ lora_A.T)
        delta_weight = (self.lora_A @ self.lora_B).T * self.scaling
        self.base_linear.weight.data += delta_weight
        self.merged = True

    def unmerge(self) -> None:
        """拆分合并后的权重（若需要继续训练或切换其他 LoRA 适配器）。"""
        if not self.merged:
            return
        delta_weight = (self.lora_A @ self.lora_B).T * self.scaling
        self.base_linear.weight.data -= delta_weight
        self.merged = False


def inject_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("query", "value"),
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """递归遍历模型，将匹配 target_modules 的 nn.Linear 替换为 LoRALinear。"""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(
                model,
                name,
                LoRALinear(module, rank=rank, alpha=alpha).to(
                    device=module.weight.device, dtype=module.weight.dtype
                ),
            )
        else:
            inject_lora(module, target_modules=target_modules, rank=rank, alpha=alpha)
    return model


class LoRAClassifier(nn.Module):
    """基于 LoRA 微调基座编码器 + 分类头的完整模型。"""

    def __init__(
        self,
        base_model_name: str,
        num_classes: int = 2,
        target_modules: tuple[str, ...] = ("query", "value"),
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # 1. 冻结基座所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 2. 在目标线性层注入 LoRA
        inject_lora(
            self.base_model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
        )

        # 3. 任务分类头（可训练）
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # 取首个 token ([CLS]) 的隐状态进行分类
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden)
        return logits

    def merge_lora(self) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge()

    def unmerge_lora(self) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.unmerge()

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """仅提取可训练参数（LoRA 权重与分类头），避免保存庞大的冻结基座。"""
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k in trainable_keys
        }


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    print(
        f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({ratio:.4f}%)"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LoRAClassifier(
        base_model_name=model_name,
        num_classes=2,
        target_modules=("query", "value"),
        rank=8,
        alpha=16,
    ).to(device)

    print_trainable_parameters(model)

    texts = [
        "LoRA adapts large models efficiently.",
        "Low-rank updates keep inference fast.",
    ]
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 仅优化 requires_grad=True 的参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = 3
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 每次迭代重新前向传播，构建独立的计算图
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 验证权重合并前后输出一致性
    model.eval()
    with torch.no_grad():
        logits_before_merge = model(input_ids=input_ids, attention_mask=attention_mask)
        model.merge_lora()
        logits_after_merge = model(input_ids=input_ids, attention_mask=attention_mask)
        max_diff = (logits_before_merge - logits_after_merge).abs().max().item()
        print(f"合并前后 Logits 最大绝对误差: {max_diff:.2e}")
        model.unmerge_lora()

    # 仅保存轻量级的 LoRA + 分类头权重
    lora_state = model.get_lora_state_dict()
    torch.save(lora_state, "lora_adapter_weights.pt")
    print(f"已保存轻量适配器权重至 lora_adapter_weights.pt，共 {len(lora_state)} 个张量。")


if __name__ == "__main__":
    main()