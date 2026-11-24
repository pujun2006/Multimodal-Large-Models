import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class AdapterModule(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super().__init__()
        self.down_projection = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        return x + self.up_projection(self.activation(self.down_projection(x)))


class PEFTModel(nn.Module):
    def __init__(self, base_model_name, adapter_dim, num_labels=2):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        for param in self.base_model.parameters():
            param.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        num_layers = self.base_model.config.num_hidden_layers
        self.adapters = nn.ModuleList(
            [AdapterModule(hidden_size, adapter_dim) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = list(outputs.hidden_states)
        # 跳过第 0 层 embedding，对各 Transformer 层隐状态应用 Adapter
        for i, adapter in enumerate(self.adapters):
            hidden_states[i + 1] = adapter(hidden_states[i + 1])
        logits = self.classifier(hidden_states[-1][:, 0, :])
        return logits


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, is_peft=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        if is_peft:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_and_eval(
    name, model, train_loader, val_loader, criterion, optimizer, device, epochs=3, is_peft=False
):
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    init_loss, init_acc = evaluate(model, val_loader, criterion, device, is_peft=is_peft)
    print(f"[{name}] 训练前验证集 -> Loss: {init_loss:.4f}, Acc: {init_acc:.2%}")

    for epoch in range(epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_t = time.perf_counter()

        train_loss_sum = 0.0
        train_count = 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if is_peft:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_t

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, is_peft=is_peft)
        print(
            f"[{name}] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss_sum / max(train_count, 1):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
            f"耗时: {elapsed:.2f}s"
        )

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[{name}] 峰值显存占用: {peak_mb:.1f} MB")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total * 100) if total else 0.0
    return total, trainable, ratio


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = [
        "I love this movie!",
        "This is a terrible product.",
        "Fantastic experience.",
        "Not worth the price.",
    ]
    train_labels = [1, 0, 1, 0]

    val_texts = [
        "Really enjoyed the story and acting.",
        "Broke after two days, very disappointed.",
    ]
    val_labels = [1, 0]

    train_loader = DataLoader(
        TextDataset(train_texts, train_labels, tokenizer), batch_size=2, shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val_texts, val_labels, tokenizer), batch_size=2, shuffle=False
    )
    criterion = nn.CrossEntropyLoss()

    set_seed(42)
    full_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer_full = torch.optim.AdamW(full_model.parameters(), lr=2e-5)
    t_tot, t_trn, t_rat = count_params(full_model)
    print(f"全参数微调参数量 -> 总计: {t_tot:,} | 可训练: {t_trn:,} ({t_rat:.2f}%)")
    train_and_eval(
        "全参数微调",
        full_model,
        train_loader,
        val_loader,
        criterion,
        optimizer_full,
        device,
        epochs=3,
        is_peft=False,
    )

    print("-" * 60)

    set_seed(42)
    peft_model = PEFTModel(model_name, adapter_dim=32, num_labels=2)
    optimizer_peft = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, peft_model.parameters()), lr=1e-3
    )
    p_tot, p_trn, p_rat = count_params(peft_model)
    print(f"PEFT微调参数量 -> 总计: {p_tot:,} | 可训练: {p_trn:,} ({p_rat:.2f}%)")
    train_and_eval(
        "PEFT微调",
        peft_model,
        train_loader,
        val_loader,
        criterion,
        optimizer_peft,
        device,
        epochs=3,
        is_peft=True,
    )