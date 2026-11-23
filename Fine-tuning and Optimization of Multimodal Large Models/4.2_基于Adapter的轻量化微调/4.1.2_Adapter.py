import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AdapterLayer(nn.Module):
    """标准瓶颈式 Adapter：降维 -> 激活 -> 升维 + 残差连接。"""

    def __init__(self, hidden_size: int, adapter_dim: int, init_std: float = 1e-3):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, hidden_size)

        nn.init.normal_(self.down_proj.weight, std=init_std)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_std)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up_proj(self.act(self.down_proj(x)))


def _make_adapter_hook(adapter: AdapterLayer):
    """生成前向钩子：若模块返回 tuple，则修改首个隐状态张量后拼回。"""

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            adapted_first = adapter(output[0])
            return (adapted_first,) + output[1:]
        return adapter(output)

    return hook


class AdapterClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, adapter_dim: int = 64):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)

        # 1. 冻结基座参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        self.attn_adapters = nn.ModuleList()
        self.ffn_adapters = nn.ModuleList()

        # 2. 为每一层的 attention 和 output 注册 Adapter 钩子
        for layer in self.base_model.encoder.layer:
            attn_adapter = AdapterLayer(hidden_size, adapter_dim)
            ffn_adapter = AdapterLayer(hidden_size, adapter_dim)
            self.attn_adapters.append(attn_adapter)
            self.ffn_adapters.append(ffn_adapter)

            layer.attention.register_forward_hook(_make_adapter_hook(attn_adapter))
            layer.output.register_forward_hook(_make_adapter_hook(ffn_adapter))

        # 3. 任务分类头（默认可训练）
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_repr)

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k in trainable_keys
        }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AdapterClassifier(model_name, num_classes=2, adapter_dim=64).to(device)

    texts = ["This is a positive example.", "This is a negative example."]
    labels = torch.tensor([1, 0], dtype=torch.long, device=device)
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    torch.save(model.get_trainable_state_dict(), "adapter_weights.pt")
    print("已保存可训练的 Adapter 与分类头权重。")


if __name__ == "__main__":
    main()
