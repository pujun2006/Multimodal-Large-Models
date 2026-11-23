import gc
import os

# 缓解 CUDA 显存碎片（需在 import torch 前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_gpu_memory() -> None:
    """清理可能残留的 Python 对象与 CUDA 缓存"""
    for var_name in ["model", "optimizer", "train_loader", "val_loader", "outputs", "loss"]:
        if var_name in globals():
            del globals()[var_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalInstructionDataset(Dataset):
    def __init__(self, instructions: list[str], targets: list[str], tokenizer, max_length: int = 256):
        assert len(instructions) == len(targets)
        self.samples = []
        eos_id = tokenizer.eos_token_id

        for inst, tgt in zip(instructions, targets):
            prompt_text = f"指令：{inst}\n回答："
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
            target_ids = tokenizer(tgt, truncation=True, max_length=max_length)["input_ids"]

            input_ids = prompt_ids + target_ids
            if eos_id is not None and (len(input_ids) == 0 or input_ids[-1] != eos_id):
                input_ids = input_ids + [eos_id]

            input_ids = input_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(input_ids))

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            attention_mask = [1] * len(input_ids)

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids_list, mask_list, labels_list = [], [], []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids_list.append(item["input_ids"] + [pad_token_id] * pad_len)
            mask_list.append(item["attention_mask"] + [0] * pad_len)
            labels_list.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    return collate_fn


def freeze_except_top_layers(model, num_top_layers: int = 2) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers[-num_top_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数量: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")


def evaluate_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += outputs.loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def generate_predictions(model, tokenizer, prompts: list[str], device: torch.device) -> list[str]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for inst in prompts:
            prompt_text = f"指令：{inst}\n回答："
            encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_len = encoded["input_ids"].shape[1]

            output_ids = model.generate(
                **encoded,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = output_ids[0][input_len:]
            predictions.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return predictions


def main() -> None:
    clean_gpu_memory()
    set_seed(42)

    train_instructions = [
        "将以下句子翻译成中文：This is a test sentence.",
        "将以下句子翻译成中文：Machine learning is useful.",
        "将下面的数字序列从小到大排序：5, 2, 8, 1.",
        "将下面的数字序列从小到大排序：12, 4, 9, 3.",
        "为以下问题生成简明答案：什么是多模态大模型？",
        "为以下问题生成简明答案：什么是指令微调？",
    ]
    train_targets = [
        "这是一个测试句子。",
        "机器学习很有用。",
        "1, 2, 5, 8.",
        "3, 4, 9, 12.",
        "多模态大模型是一种能够融合文本、图像等多种模态信息进行理解与生成的模型。",
        "指令微调是使用指令与目标回复数据对模型进行监督微调，以提升其遵循指令的能力。",
    ]

    val_instructions = [
        "将以下句子翻译成中文：How are you?",
        "将下面的数字序列从小到大排序：7, 3, 10, 1.",
        "为以下问题生成简明答案：什么是提示学习？",
    ]
    val_targets = [
        "你好吗？",
        "1, 3, 7, 10.",
        "提示学习是通过设计提示模板或软提示引导模型完成下游任务的方法。",
    ]

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    freeze_except_top_layers(model, num_top_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        CausalInstructionDataset(train_instructions, train_targets, tokenizer),
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        CausalInstructionDataset(val_instructions, val_targets, tokenizer),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    initial_val_loss = evaluate_loss(model, val_loader, device)
    before_preds = generate_predictions(model, tokenizer, val_instructions, device)
    print(f"初始验证集 Loss: {initial_val_loss:.4f}\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=5e-5)
    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = evaluate_loss(model, val_loader, device)
        print(
            f"Epoch [{epoch}/{epochs}] - "
            f"Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}"
        )

    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n峰值已分配显存: {peak_gib:.2f} GiB")

    after_preds = generate_predictions(model, tokenizer, val_instructions, device)

    print("\n=== 微调前后效果对比 ===")
    for prompt, expected, before, after in zip(
        val_instructions, val_targets, before_preds, after_preds
    ):
        print(f"指令: {prompt}")
        print(f"参考目标: {expected}")
        print(f"微调前输出: {before}")
        print(f"微调后输出: {after}")
        print("-" * 50)


if __name__ == "__main__":
    main()