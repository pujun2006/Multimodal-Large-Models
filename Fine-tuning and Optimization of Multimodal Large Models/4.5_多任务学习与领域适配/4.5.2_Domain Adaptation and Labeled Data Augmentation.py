import os
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


# ==========================================
# 1. 随机种子与设备配置
# ==========================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # 领域分类模型（也可用 "bert-base-uncased"）
    cls_model_name: str = "distilbert-base-uncased"
    # 领域持续预训练（CPT）模型（若离线环境无因果语言模型，可设 enable_cpt=False）
    enable_cpt: bool = False
    cpt_model_name: str = "gpt2"

    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 4
    cpt_epochs: int = 2
    cpt_lr: float = 5e-5
    sft_epochs: int = 5
    sft_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    output_dir: str = "./domain_adapted_artifacts"


# ==========================================
# 2. 标注数据增强（带受保护词与标点/词元处理）
# ==========================================
class LabelPreservingAugmentor:
    """
    针对标注文本的数据增强器：
    1. 保留词元边界与标点（避免 "weather." 无法匹配 "weather"）
    2. 受保护词（protected_terms）不被替换，防止实体/关键领域词语义漂移
    3. 支持同义词替换 + 局部词序微扰（可选）
    """

    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        synonyms: Dict[str, List[str]],
        protected_terms: List[str] = None,
        replace_prob: float = 0.6,
    ):
        # 将词典 key 统一转小写便于匹配，同时保留原大小写风格
        self.synonyms = {k.lower(): v for k, v in synonyms.items()}
        self.protected_terms = {t.lower() for t in (protected_terms or [])}
        self.replace_prob = replace_prob

    @staticmethod
    def _match_case(src_word: str, target_word: str) -> str:
        if src_word.istitle():
            return target_word.title()
        if src_word.isupper():
            return target_word.upper()
        return target_word

    @staticmethod
    def _detokenize(tokens: List[str]) -> str:
        text = ""
        for tok in tokens:
            if not text:
                text = tok
            elif re.match(r"^[^\w\s]$", tok):
                # 标点符号紧跟前词（如句号、逗号、撇号）
                text += tok
            elif text.endswith("'") or tok == "'":
                text += tok
            else:
                text += " " + tok
        return text

    def augment_text(self, text: str) -> str:
        tokens = self.TOKEN_PATTERN.findall(text)
        new_tokens = []

        for tok in tokens:
            low = tok.lower()
            if (
                re.match(r"^\w+$", tok)
                and low not in self.protected_terms
                and low in self.synonyms
                and random.random() < self.replace_prob
            ):
                candidate = random.choice(self.synonyms[low])
                new_tokens.append(self._match_case(tok, candidate))
            else:
                new_tokens.append(tok)

        augmented = self._detokenize(new_tokens)
        # 若未发生任何替换，返回原句（上层调用方可决定是否去重）
        return augmented

    def augment_dataset(
        self,
        texts: List[str],
        labels: List[int],
        num_aug_per_sample: int = 2,
    ) -> Tuple[List[str], List[int]]:
        """
        仅应在划分出训练集（train split）后调用，严禁对验证集/测试集做增强！
        """
        aug_texts: List[str] = []
        aug_labels: List[int] = []

        for text, label in zip(texts, labels):
            # 1. 保留原始高质量标注样本
            aug_texts.append(text)
            aug_labels.append(label)

            # 2. 生成增强样本并过滤完全重复样本
            seen_variants = {text.strip().lower()}
            for _ in range(num_aug_per_sample):
                variant = self.augment_text(text)
                norm_variant = variant.strip().lower()
                if norm_variant not in seen_variants:
                    seen_variants.add(norm_variant)
                    aug_texts.append(variant)
                    aug_labels.append(label)

        return aug_texts, aug_labels


# ==========================================
# 3. PyTorch Dataset 封装
# ==========================================
class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ==========================================
# 4. 阶段一（可选）：无监督领域持续预训练（CPT）
# ==========================================
def continual_pretrain(domain_corpus: List[str], cfg: Config, device: torch.device) -> str:
    """
    使用领域内未标注文本做自监督语言建模，内化领域词汇与行文分布。
    返回保存后的 CPT 权重目录路径。
    """
    cpt_dir = os.path.join(cfg.output_dir, "cpt_model")
    os.makedirs(cpt_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.cpt_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.cpt_model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    encodings = tokenizer(
        domain_corpus,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
        return_tensors="pt",
    )

    class CausalCorpusDataset(Dataset):
        def __len__(self):
            return encodings["input_ids"].size(0)

        def __getitem__(self, idx):
            return {
                "input_ids": encodings["input_ids"][idx],
                "attention_mask": encodings["attention_mask"][idx],
            }

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(CausalCorpusDataset(), batch_size=2, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.cpt_lr, weight_decay=cfg.weight_decay)
    model.train()

    for epoch in range(cfg.cpt_epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[CPT] Epoch {epoch + 1}/{cfg.cpt_epochs} | Loss: {total_loss / len(loader):.4f}")

    model.save_pretrained(cpt_dir)
    tokenizer.save_pretrained(cpt_dir)
    return cpt_dir


# ==========================================
# 5. 领域分类模型与训练/评估循环
# ==========================================
class DomainClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_sft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(cfg.output_dir, "best_domain_classifier.pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.sft_lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.sft_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_acc = -1.0

    for epoch in range(cfg.sft_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"[SFT] Epoch {epoch + 1}/{cfg.sft_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                },
                best_ckpt_path,
            )

    print(f"最佳模型已保存至: {best_ckpt_path} (最佳验证集准确率: {best_val_acc:.4f})")
    return best_ckpt_path


# ==========================================
# 6. 推理演示
# ==========================================
def predict(
    texts: List[str],
    tokenizer,
    ckpt_path: str,
    cfg: Config,
    device: torch.device,
    label_names: Dict[int, str],
) -> None:
    model = DomainClassifier(cfg.cls_model_name, num_labels=cfg.num_labels)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()

    print("\n=== 推理结果 ===")
    for text, pred_id, prob_row in zip(texts, pred_ids, probs.tolist()):
        pred_label = label_names.get(pred_id, str(pred_id))
        print(f"文本: {text}\n预测类别: {pred_label} | 各类别概率: {[round(p, 4) for p in prob_row]}\n")


# ==========================================
# 7. 主流程入口
# ==========================================
def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    label_names = {0: "财经/商业 (Finance)", 1: "科技/研发 (Tech)"}

    # 可选：未标注领域语料（用于 CPT 持续预训练）
    unlabeled_domain_corpus = [
        "Quarterly earnings guidance and bond yields shifted after the central bank statement.",
        "Distributed transformer training and low-rank adaptation reduce GPU memory overhead.",
        "Venture capital funding in enterprise AI infrastructure expanded across major markets.",
        "Vector indexing and retrieval pipelines improve factual grounding in domain assistants.",
    ]
    if cfg.enable_cpt:
        continual_pretrain(unlabeled_domain_corpus, cfg, device)

    # 原始标注样本（示例扩充至 8 条，演示先划分再增强）
    raw_texts = [
        "The quarterly market report and inflation outlook are very detailed.",
        "Central bank interest rate policy strongly affects equity valuations.",
        "Corporate bond yields rose following the new macroeconomic forecast.",
        "Institutional investors rebalanced portfolios ahead of earnings season.",
        "The new transformer architecture and inference engine are revolutionary.",
        "Quantized neural networks accelerate low-latency model deployment on edge devices.",
        "Distributed training clusters optimize gradient synchronization across nodes.",
        "Vector search and retrieval pipelines enhance domain knowledge grounding.",
    ]
    raw_labels = [0, 0, 0, 0, 1, 1, 1, 1]

    # 1. 先切分训练集与验证集（严格防止数据泄漏）
    indices = list(range(len(raw_texts)))
    random.shuffle(indices)
    split_idx = int(len(indices) * 0.75)  # 6 条训练，2 条验证
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_texts = [raw_texts[i] for i in train_idx]
    train_labels = [raw_labels[i] for i in train_idx]
    val_texts = [raw_texts[i] for i in val_idx]
    val_labels = [raw_labels[i] for i in val_idx]

    # 2. 仅对训练集执行标注数据增强
    synonyms = {
        "market": ["economy", "financial sector"],
        "report": ["analysis", "briefing"],
        "detailed": ["comprehensive", "thorough"],
        "policy": ["framework", "strategy"],
        "new": ["novel", "latest"],
        "revolutionary": ["transformative", "groundbreaking"],
        "accelerate": ["speed up", "boost"],
        "enhance": ["improve", "strengthen"],
    }
    protected_terms = ["transformer", "quantized", "bond", "inflation"]

    augmentor = LabelPreservingAugmentor(
        synonyms=synonyms,
        protected_terms=protected_terms,
        replace_prob=0.65,
    )
    aug_train_texts, aug_train_labels = augmentor.augment_dataset(
        train_texts, train_labels, num_aug_per_sample=2
    )
    print(f"原始训练集样本数: {len(train_texts)} -> 增强后训练集样本数: {len(aug_train_texts)}")

    # 3. 构建 Tokenizer、Dataset 与 DataLoader
    tokenizer = AutoTokenizer.from_pretrained(cfg.cls_model_name)
    train_dataset = TextClassificationDataset(aug_train_texts, aug_train_labels, tokenizer, cfg.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4. 初始化并训练领域适配模型
    model = DomainClassifier(cfg.cls_model_name, num_labels=cfg.num_labels).to(device)
    best_ckpt = train_sft(model, train_loader, val_loader, cfg, device)

    # 5. 新样本推理验证
    test_samples = [
        "The latest market analysis shows bond yields stabilizing.",
        "A novel transformer model speeds up low-latency inference.",
    ]
    predict(test_samples, tokenizer, best_ckpt, cfg, device, label_names)


if __name__ == "__main__":
    main()
