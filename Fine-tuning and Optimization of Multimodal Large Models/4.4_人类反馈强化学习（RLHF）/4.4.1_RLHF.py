import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 默认没有 pad_token，用 eos_token 作为 pad_token 方便批处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ==========================================
# 一、监督微调（SFT）
# ==========================================
class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.items = []
        for prompt, response in samples:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            resp_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
            input_ids = (prompt_ids + resp_ids)[:max_length]
            # prompt 部分设 label 为 -100，仅监督模型生成的回答部分
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            attention_mask = [1] * len(input_ids)

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attention_mask += [0] * pad_len

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_sft(model, tokenizer, sft_samples, device, epochs=2, lr=5e-5, batch_size=2):
    dataset = SFTDataset(sft_samples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[SFT] Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")


# ==========================================
# 二、奖励模型（Reward Model - 成对偏好）
# ==========================================
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = copy.deepcopy(base_model)
        # 去掉语言模型词表投影头，替换为输出标量奖励的线性头
        if hasattr(self.backbone, "lm_head"):
            self.backbone.lm_head = nn.Identity()
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        # 取每个样本最后一个有效非 padding token 的隐藏状态
        seq_lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_indices, seq_lengths]  # [B, H]
        rewards = self.score_head(pooled).squeeze(-1)  # [B]
        return rewards


class PreferenceDataset(Dataset):
    def __init__(self, pref_samples, tokenizer, max_length=128):
        self.items = []
        for prompt, chosen, rejected in pref_samples:
            c_enc = tokenizer(
                prompt + chosen + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            r_enc = tokenizer(
                prompt + rejected + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.items.append({
                "chosen_ids": c_enc["input_ids"].squeeze(0),
                "chosen_mask": c_enc["attention_mask"].squeeze(0),
                "rejected_ids": r_enc["input_ids"].squeeze(0),
                "rejected_mask": r_enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_reward_model(rm, tokenizer, pref_samples, device, epochs=3, lr=2e-5, batch_size=2):
    dataset = PreferenceDataset(pref_samples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr)

    rm.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            optimizer.zero_grad()
            r_chosen = rm(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
            r_rejected = rm(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))

            # Bradley-Terry 偏好损失: -log(sigmoid(r_w - r_l)) = softplus(-(r_w - r_l))
            loss = F.softplus(-(r_chosen - r_rejected)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.size(0)

        acc = correct / max(total, 1)
        print(f"[RM]  Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f} | Pair Acc: {acc:.2%}")


# ==========================================
# 三、强化学习微调（带 KL 惩罚的策略梯度）
# ==========================================
def compute_response_logprob_sum(model, full_ids, attention_mask, prompt_lengths):
    """
    计算每个样本回答部分（prompt 之后、非 padding 部分）的 token log_prob 之和。
    full_ids: [B, T], prompt_lengths: [B]
    """
    outputs = model(input_ids=full_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, :-1, :]  # [B, T-1, V]
    targets = full_ids[:, 1:]           # [B, T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # 构造回答区域的有效 mask（排除 prompt 部分与 padding 部分）
    seq_len = token_log_probs.size(1)
    positions = torch.arange(seq_len, device=full_ids.device).unsqueeze(0)  # [1, T-1]
    # full_ids[:, 1:] 对应的原序列索引是 1..T-1，因此回答起始位置在 targets 中为 prompt_len - 1
    start_idx = (prompt_lengths - 1).clamp(min=0).unsqueeze(1)  # [B, 1]
    is_response = positions >= start_idx
    not_pad = attention_mask[:, 1:].bool()
    resp_mask = (is_response & not_pad).float()

    return (token_log_probs * resp_mask).sum(dim=-1)  # [B]


def train_rl_policy(policy_model, ref_model, rm, tokenizer, prompts, device, epochs=3, lr=1e-5, kl_beta=0.05):
    ref_model.eval()
    rm.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    for p in rm.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    # 左填充更适合因果模型批量生成
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for epoch in range(epochs):
        policy_model.train()
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        prompt_ids = enc["input_ids"].to(device)
        prompt_mask = enc["attention_mask"].to(device)
        # 左填充下，每个样本的 prompt 有效长度均为 prompt_ids.size(1)（若未截断）
        prompt_lengths = torch.full((prompt_ids.size(0),), prompt_ids.size(1), device=device, dtype=torch.long)

        # 1. 采样生成回答
        policy_model.eval()
        with torch.no_grad():
            gen_ids = policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=24,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_mask = (gen_ids != tokenizer.pad_token_id).long()
            full_mask[:, :prompt_ids.size(1)] = prompt_mask

            rm_scores = rm(gen_ids, full_mask)  # [B]
            ref_resp_logp = compute_response_logprob_sum(ref_model, gen_ids, full_mask, prompt_lengths)

        # 2. 前向计算当前策略回答对数概率（带梯度）
        policy_model.train()
        optimizer.zero_grad()
        policy_resp_logp = compute_response_logprob_sum(policy_model, gen_ids, full_mask, prompt_lengths)

        with torch.no_grad():
            # 序列级 KL 近似: E[log(pi / pi_ref)]
            kl = (policy_resp_logp - ref_resp_logp).clamp(min=-20.0, max=20.0)
            shaped_reward = rm_scores - kl_beta * kl
            # 减去批次均值作为基线，降低策略梯度方差
            advantage = shaped_reward - shaped_reward.mean()
            if advantage.std() > 1e-6:
                advantage = advantage / (advantage.std() + 1e-8)

        # 策略梯度损失：最小化 -advantage * log_prob
        pg_loss = -(advantage * policy_resp_logp).mean()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        print(
            f"[RL]  Epoch {epoch + 1}/{epochs} | Loss: {pg_loss.item():.4f} | "
            f"Mean RM: {rm_scores.mean().item():.4f} | Mean KL: {kl.mean().item():.4f}"
        )

    tokenizer.padding_side = orig_padding_side


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    model_name = "gpt2"
    tokenizer = setup_tokenizer(model_name)

    sft_data = [
        ("Question: What is overfitting in machine learning?\nAnswer: ",
         "Overfitting happens when a model memorizes training data and performs poorly on unseen data."),
        ("Question: How can I stay focused while studying?\nAnswer: ",
         "Use short focused work blocks, silence notifications, and review key points actively."),
    ]

    pref_data = [
        (
            "Question: Explain gradient descent simply.\nAnswer: ",
            "Gradient descent iteratively adjusts parameters in the opposite direction of the gradient to minimize loss.",
            "Gradient descent is just a random guess that sometimes works.",
        ),
        (
            "Question: Why is validation data important?\nAnswer: ",
            "Validation data helps detect overfitting and tune hyperparameters on data not used for parameter updates.",
            "You do not need validation data if your training accuracy is high.",
        ),
    ]

    rl_prompts = [
        "Question: What is a neural network?\nAnswer: ",
        "Question: How do you debug a slow program?\nAnswer: ",
    ]

    # 1. 初始化并训练 SFT 模型
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    train_sft(policy_model, tokenizer, sft_data, device, epochs=2)

    # 2. 保存 SFT 权重作为参考模型（Reference Model）与奖励模型底座
    ref_model = copy.deepcopy(policy_model).to(device)
    reward_model = RewardModel(policy_model).to(device)

    # 3. 训练奖励模型
    train_reward_model(reward_model, tokenizer, pref_data, device, epochs=3)

    # 4. 强化学习优化策略模型
    train_rl_policy(policy_model, ref_model, reward_model, tokenizer, rl_prompts, device, epochs=3)

    # 5. 验证对齐后的生成效果
    policy_model.eval()
    test_prompt = "Question: What is a neural network?\nAnswer: "
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = policy_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("\n=== 生成测试 ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
