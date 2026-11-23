import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def set_seed(seed: int = 42) -> None:
    """固定随机种子，提升小样本实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(42)

    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载分词器与预训练序列分类模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # 2. 配置 LoRA（小样本场景下使用较小秩 r 与适当 dropout 抑制过拟合）
    # BERT 注意力层线性映射名称通常为 "query" 和 "value"
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    # 3. 构造小样本支持集（每类 4 条，共 8 条样本；标签 1=正面, 0=负面）
    train_texts = [
        "This product works smoothly and exceeded my expectations.",
        "Really happy with the build quality and fast delivery.",
        "Great value for money, I use it every single day.",
        "Super intuitive interface and very reliable performance.",
        "Terrible experience, the item stopped working after two days.",
        "I do not like this product at all, total waste of money.",
        "Poor customer support and confusing setup instructions.",
        "Disappointed with the quality, it feels cheap and fragile.",
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0]

    encoded = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 仅将 requires_grad=True 的参数（LoRA 旁路参数与分类头）传入优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 5. 小样本微调循环
    epochs = 6
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2%}")

    # 6. 推理测试
    test_texts = [
        "I love this item, it makes my workflow so much faster!",
        "Awful quality, completely regret buying it.",
        "Pretty solid overall, no major complaints.",
        "Not worth the price, broke on the first try.",
    ]
    test_inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(**test_inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    label_map = {0: "Negative (0)", 1: "Positive (1)"}
    print("\n--- 测试集预测结果 ---")
    for text, pred, prob in zip(test_texts, predictions.cpu().tolist(), probs.cpu().tolist()):
        conf = prob[pred]
        print(f"[{label_map[pred]} | 置信度 {conf:.2%}] {text}")


if __name__ == "__main__":
    main()