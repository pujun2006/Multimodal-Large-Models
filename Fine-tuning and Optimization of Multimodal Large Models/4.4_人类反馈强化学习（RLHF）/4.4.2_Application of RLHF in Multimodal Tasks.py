import copy
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_image_tensor(image_path: str) -> torch.Tensor:
    """
    返回形状为 [3, 224, 224] 的图像张量。
    若本地路径不存在，则生成一张确定性的合成 RGB 图像，确保示例代码可直接运行。
    """
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        return transform(img)

    # 离线/无图环境回退：根据路径生成稳定伪随机图像
    seed_val = abs(hash(image_path or "default_img")) % (2**31)
    rng = torch.Generator().manual_seed(seed_val)
    fake_uint8 = torch.randint(0, 256, (224, 224, 3), generator=rng, dtype=torch.uint8)
    img = Image.fromarray(fake_uint8.numpy(), mode="RGB")
    return transform(img)


class MultiModalVisionEncoder(nn.Module):
    """将图像编码为与语言模型隐藏维度对齐的视觉特征向量 [B, D_model]。"""

    def __init__(self, hidden_size: int, use_pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出 [B, 2048, 1, 1]
        self.projector = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 224, 224]
        feat = self.backbone(images).flatten(1)  # [B, 2048] 避免 batch=1 时 squeeze 掉 batch 维
        return self.projector(feat)  # [B, hidden_size]


class MultiModalPolicyLM(nn.Module):
    """
    在文本序列最前端插入 1 个视觉 token（形状 [B, 1, H]），
    让后续文本 token 在因果自注意力中能够 attend 到视觉表征。
    """

    def __init__(self, text_model_name: str, use_pretrained_vision: bool = True):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(text_model_name)
        hidden_size = self.lm.config.hidden_size
        self.vision_encoder = MultiModalVisionEncoder(
            hidden_size=hidden_size,
            use_pretrained=use_pretrained_vision,
        )

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def _build_mm_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor):
        """
        拼接视觉 token 与文本 token embeddings：
        inputs_embeds: [B, 1 + T, H]
        mm_mask:       [B, 1 + T]
        """
        vis_emb = self.vision_encoder(images).unsqueeze(1)  # [B, 1, H]
        txt_emb = self.lm.get_input_embeddings()(input_ids)  # [B, T, H]
        inputs_embeds = torch.cat([vis_emb, txt_emb], dim=1)

        vis_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
        mm_mask = torch.cat([vis_mask, attention_mask], dim=1)
        return inputs_embeds, mm_mask

    def forward_sft(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, images: torch.Tensor):
        inputs_embeds, mm_mask = self._build_mm_inputs(input_ids, attention_mask, images)
        # 在标签最前方补 -100，对应视觉 token 位置不计算语言建模 loss
        vis_ignore = torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype)
        mm_labels = torch.cat([vis_ignore, labels], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=mm_mask, labels=mm_labels, return_dict=True)

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        返回与 input_ids 对齐的文本 logits，形状为 [B, T, V]。
        """
        inputs_embeds, mm_mask = self._build_mm_inputs(input_ids, attention_mask, images)
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=mm_mask, return_dict=True)
        # 去掉第 0 个视觉 token 对应的输出位，保留 [B, T, V]
        return outputs.logits[:, 1:, :]

    @torch.no_grad()
    def generate_response(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        images: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 24,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        基于多模态前缀（视觉 token + 左填充 prompt）自回归采样生成回答，
        返回完整文本序列 [B, T_prompt + T_gen]（未生成满时右侧补 pad_token_id）。
        """
        self.eval()
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        vis_emb = self.vision_encoder(images).unsqueeze(1)  # [B, 1, H]
        cur_embeds = torch.cat([vis_emb, self.lm.get_input_embeddings()(prompt_ids)], dim=1)  # [B, 1 + T, H]
        cur_mask = torch.cat([torch.ones((batch_size, 1), device=device, dtype=prompt_mask.dtype), prompt_mask], dim=1)

        generated_tokens = torch.full((batch_size, max_new_tokens), pad_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for step in range(max_new_tokens):
            outputs = self.lm(inputs_embeds=cur_embeds, attention_mask=cur_mask, return_dict=True)
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)

            # Top-p (nucleus) 采样
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_sorted_idx).squeeze(-1)  # [B]

            next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)
            generated_tokens[:, step] = next_token

            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

            next_emb = self.lm.get_input_embeddings()(next_token).unsqueeze(1)  # [B, 1, H]
            next_step_mask = (~finished).long().unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            cur_mask = torch.cat([cur_mask, next_step_mask], dim=1)

        return torch.cat([prompt_ids, generated_tokens], dim=1)


# ==========================================
# 一、多模态监督微调（Multimodal SFT）
# ==========================================
class MultimodalSFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.items = []
        for image_path, prompt, response in samples:
            image_tensor = load_image_tensor(image_path)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            resp_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
            input_ids = (prompt_ids + resp_ids)[:max_length]

            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            attention_mask = [1] * len(input_ids)

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attention_mask += [0] * pad_len

            self.items.append({
                "images": image_tensor,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_multimodal_sft(model, tokenizer, sft_samples, device, epochs=2, lr=5e-5, batch_size=2):
    dataset = MultimodalSFTDataset(sft_samples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model.forward_sft(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                images=batch["images"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[MM-SFT] Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")


# ==========================================
# 二、多模态奖励模型（成对偏好 + Bradley-Terry）
# ==========================================
class MultiModalRewardModel(nn.Module):
    def __init__(self, base_mm_model: MultiModalPolicyLM):
        super().__init__()
        self.mm_encoder = copy.deepcopy(base_mm_model)
        # 去掉语言模型词表投影头，仅保留隐藏状态提取能力
        if hasattr(self.mm_encoder.lm, "lm_head"):
            self.mm_encoder.lm.lm_head = nn.Identity()
        hidden_size = self.mm_encoder.lm.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        vis_emb = self.mm_encoder.vision_encoder(images).unsqueeze(1)  # [B, 1, H]
        txt_emb = self.mm_encoder.lm.get_input_embeddings()(input_ids)  # [B, T, H]
        inputs_embeds = torch.cat([vis_emb, txt_emb], dim=1)

        vis_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
        mm_mask = torch.cat([vis_mask, attention_mask], dim=1)

        outputs = self.mm_encoder.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=mm_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, 1 + T, H]
        # 取最后一个有效非 padding token 的隐藏状态
        last_idx = mm_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_idx, last_idx]  # [B, H]
        return self.score_head(pooled).squeeze(-1)  # [B]


class MultimodalPreferenceDataset(Dataset):
    def __init__(self, pref_samples, tokenizer, max_length=128):
        self.items = []
        for image_path, prompt, chosen, rejected in pref_samples:
            image_tensor = load_image_tensor(image_path)
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
                "images": image_tensor,
                "chosen_ids": c_enc["input_ids"].squeeze(0),
                "chosen_mask": c_enc["attention_mask"].squeeze(0),
                "rejected_ids": r_enc["input_ids"].squeeze(0),
                "rejected_mask": r_enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_multimodal_reward_model(rm, tokenizer, pref_samples, device, epochs=3, lr=2e-5, batch_size=2):
    dataset = MultimodalPreferenceDataset(pref_samples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr)

    rm.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            optimizer.zero_grad()
            images = batch["images"].to(device)
            r_chosen = rm(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device), images)
            r_rejected = rm(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device), images)

            loss = F.softplus(-(r_chosen - r_rejected)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.size(0)

        acc = correct / max(total, 1)
        print(f"[MM-RM]  Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f} | Pair Acc: {acc:.2%}")


# ==========================================
# 三、多模态强化学习微调（带 KL 惩罚的策略梯度）
# ==========================================
def compute_response_logprob_sum(model: MultiModalPolicyLM, full_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
    logits = model.forward_logits(full_ids, attention_mask, images)  # [B, T, V]
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    targets = full_ids[:, 1:]         # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    seq_len = token_log_probs.size(1)
    positions = torch.arange(seq_len, device=full_ids.device).unsqueeze(0)
    start_idx = (prompt_lengths - 1).clamp(min=0).unsqueeze(1)
    is_response = positions >= start_idx
    not_pad = attention_mask[:, 1:].bool()
    resp_mask = (is_response & not_pad).float()

    return (token_log_probs * resp_mask).sum(dim=-1)  # [B]


def train_multimodal_rl_policy(policy_model, ref_model, rm, tokenizer, rl_samples, device, epochs=3, lr=1e-5, kl_beta=0.05):
    ref_model.eval()
    rm.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    for p in rm.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    images = torch.stack([load_image_tensor(img_path) for img_path, _ in rl_samples], dim=0).to(device)
    prompts = [prompt for _, prompt in rl_samples]

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for epoch in range(epochs):
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        prompt_ids = enc["input_ids"].to(device)
        prompt_mask = enc["attention_mask"].to(device)
        prompt_lengths = torch.full((prompt_ids.size(0),), prompt_ids.size(1), device=device, dtype=torch.long)

        # 1. 多模态采样生成
        gen_ids = policy_model.generate_response(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            images=images,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=24,
            temperature=0.8,
            top_p=0.9,
        )
        full_mask = (gen_ids != tokenizer.pad_token_id).long()
        full_mask[:, :prompt_ids.size(1)] = prompt_mask

        with torch.no_grad():
            rm_scores = rm(gen_ids, full_mask, images)
            ref_resp_logp = compute_response_logprob_sum(ref_model, gen_ids, full_mask, images, prompt_lengths)

        # 2. 策略梯度更新
        policy_model.train()
        optimizer.zero_grad()
        policy_resp_logp = compute_response_logprob_sum(policy_model, gen_ids, full_mask, images, prompt_lengths)

        with torch.no_grad():
            kl = (policy_resp_logp - ref_resp_logp).clamp(min=-20.0, max=20.0)
            shaped_reward = rm_scores - kl_beta * kl
            advantage = shaped_reward - shaped_reward.mean()
            if advantage.std() > 1e-6:
                advantage = advantage / (advantage.std() + 1e-8)

        pg_loss = -(advantage * policy_resp_logp).mean()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        print(
            f"[MM-RL]  Epoch {epoch + 1}/{epochs} | Loss: {pg_loss.item():.4f} | "
            f"Mean RM: {rm_scores.mean().item():.4f} | Mean KL: {kl.mean().item():.4f}"
        )

    tokenizer.padding_side = orig_padding_side


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    model_name = "gpt2"
    tokenizer = setup_tokenizer(model_name)

    sft_samples = [
        ("sample_1.jpg", "Image prompt: Describe the scene concisely.\nAnswer: ", "A grounded description matching the visual content."),
        ("sample_2.jpg", "Image prompt: Identify the main object.\nAnswer: ", "Focus only on elements clearly visible in the image."),
    ]

    pref_samples = [
        (
            "sample_1.jpg",
            "Image prompt: Describe the scene concisely.\nAnswer: ",
            "A concise description accurately tied to the observed image features.",
            "Inventing many unverified details that do not appear in the image at all.",
        ),
        (
            "sample_2.jpg",
            "Image prompt: Identify the main object.\nAnswer: ",
            "State the observed object clearly without fabricated attributes.",
            "Guessing random objects and colors with confident exaggeration.",
        ),
    ]

    rl_samples = [
        ("sample_3.jpg", "Image prompt: Summarize what is shown.\nAnswer: "),
        ("sample_4.jpg", "Image prompt: List key visual details.\nAnswer: "),
    ]

    # 1. 多模态 SFT
    policy_model = MultiModalPolicyLM(model_name, use_pretrained_vision=True).to(device)
    train_multimodal_sft(policy_model, tokenizer, sft_samples, device, epochs=2)

    # 2. 保存参考策略并训练多模态偏好奖励模型
    ref_model = copy.deepcopy(policy_model).to(device)
    reward_model = MultiModalRewardModel(policy_model).to(device)
    train_multimodal_reward_model(reward_model, tokenizer, pref_samples, device, epochs=3)

    # 3. 带 KL 惩罚的多模态 RL 微调
    train_multimodal_rl_policy(policy_model, ref_model, reward_model, tokenizer, rl_samples, device, epochs=3)

    # 4. 生成测试
    test_img = load_image_tensor("sample_3.jpg").unsqueeze(0).to(device)
    test_prompt = "Image prompt: Summarize what is shown.\nAnswer: "
    enc_test = tokenizer(test_prompt, return_tensors="pt").to(device)
    out_ids = policy_model.generate_response(
        prompt_ids=enc_test["input_ids"],
        prompt_mask=enc_test["attention_mask"],
        images=test_img,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=24,
        temperature=0.7,
        top_p=0.9,
    )
    print("\n=== 多模态生成测试 ===")
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
