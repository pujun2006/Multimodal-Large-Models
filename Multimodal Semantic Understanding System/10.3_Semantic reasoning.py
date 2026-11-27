from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class SemanticReasoningDecoder(nn.Module):
    """基于融合特征前缀注入的 GPT-2 语义推理与文本生成模块"""

    def __init__(
        self,
        fusion_dim: int = 512,
        gpt2_name: str = "gpt2",
        num_prefix_tokens: int = 4,
        freeze_gpt2: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_name)
        self.lm_model = GPT2LMHeadModel.from_pretrained(gpt2_name)

        # GPT-2 默认没有 pad_token，复用 eos_token 便于批处理对齐
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model.config.pad_token_id = self.tokenizer.eos_token_id

        self.embed_dim = self.lm_model.config.n_embd  # gpt2 默认为 768

        # 将融合特征 [B, fusion_dim] 映射为 [B, num_prefix_tokens * embed_dim]
        self.prefix_projector = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dim * num_prefix_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.embed_dim * num_prefix_tokens,
                self.embed_dim * num_prefix_tokens,
            ),
            nn.LayerNorm(self.embed_dim * num_prefix_tokens),
        )

        if freeze_gpt2:
            for param in self.lm_model.parameters():
                param.requires_grad = False

    def build_prefix_embeds(self, fused_features: torch.Tensor) -> torch.Tensor:
        """将多模态融合特征转换为 GPT-2 前缀词嵌入 [B, P, D]"""
        batch_size = fused_features.size(0)
        projected = self.prefix_projector(fused_features)
        return projected.view(batch_size, self.num_prefix_tokens, self.embed_dim)

    def forward(
        self,
        fused_features: torch.Tensor,
        target_texts: List[str],
        max_length: int = 64,
    ) -> torch.Tensor:
        """训练阶段前向传播：返回目标文本部分的交叉熵损失"""
        device = fused_features.device
        batch_size = fused_features.size(0)

        # 在句末追加 eos_token，教会模型在推理完成后停止生成
        formatted_texts = [
            f"{text.strip()}{self.tokenizer.eos_token}" for text in target_texts
        ]
        encoded = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = encoded["input_ids"]  # [B, L]
        attention_mask = encoded["attention_mask"]  # [B, L]

        # 获取文本词嵌入与前缀语义嵌入
        token_embeds = self.lm_model.transformer.wte(input_ids)  # [B, L, D]
        prefix_embeds = self.build_prefix_embeds(fused_features)  # [B, P, D]
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, P+L, D]

        # 前缀 token 始终可见，拼接文本 attention_mask
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 构造 labels：前缀位置设为 -100 不计入 loss；PAD 位置也设为 -100
        ignore_prefix = torch.full(
            (batch_size, self.num_prefix_tokens),
            -100,
            dtype=input_ids.dtype,
            device=device,
        )
        text_labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([ignore_prefix, text_labels], dim=1)  # [B, P+L]

        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        fused_features: torch.Tensor,
        prompt_texts: Optional[List[str]] = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.8,
    ) -> List[str]:
        """推理阶段自回归生成语义解释/结论"""
        self.eval()
        device = fused_features.device
        batch_size = fused_features.size(0)
        prefix_embeds = self.build_prefix_embeds(fused_features)  # [B, P, D]

        if prompt_texts is not None:
            enc = self.tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            prompt_ids = enc["input_ids"]
            prompt_mask = enc["attention_mask"]
            prompt_embeds = self.lm_model.transformer.wte(prompt_ids)
            cur_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prefix_mask = torch.ones(
                (batch_size, self.num_prefix_tokens),
                dtype=prompt_mask.dtype,
                device=device,
            )
            cur_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
            generated_ids: List[List[int]] = [
                seq[mask.bool()].tolist()
                for seq, mask in zip(prompt_ids, prompt_mask)
            ]
        else:
            cur_embeds = prefix_embeds
            cur_mask = torch.ones(
                (batch_size, self.num_prefix_tokens),
                dtype=torch.long,
                device=device,
            )
            generated_ids = [[] for _ in range(batch_size)]

        finished = [False] * batch_size
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            outputs = self.lm_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
            )
            next_logits = outputs.logits[:, -1, :]  # [B, V]

            if do_sample:
                scaled_logits = next_logits / max(temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B, 1]

            next_token_embeds = self.lm_model.transformer.wte(next_tokens)  # [B, 1, D]
            cur_embeds = torch.cat([cur_embeds, next_token_embeds], dim=1)
            next_mask = torch.ones(
                (batch_size, 1), dtype=cur_mask.dtype, device=device
            )
            cur_mask = torch.cat([cur_mask, next_mask], dim=1)

            for i in range(batch_size):
                if not finished[i]:
                    tok_id = next_tokens[i, 0].item()
                    generated_ids[i].append(tok_id)
                    if eos_id is not None and tok_id == eos_id:
                        finished[i] = True

            if all(finished):
                break

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟上游 ModalFusion 输出的融合特征 [batch_size=2, fusion_dim=512]
    fused_features = torch.randn(2, 512, device=device)

    # 带有语义推理/解释的目标文本
    target_reasoning_texts = [
        "The image shows a cat resting indoors on a mat, indicating a calm home environment.",
        "The image shows a dog moving outdoors in a park, suggesting active play.",
    ]

    decoder = SemanticReasoningDecoder(
        fusion_dim=512,
        gpt2_name="gpt2",
        num_prefix_tokens=4,
        freeze_gpt2=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, decoder.parameters()), lr=5e-5
    )

    # 1. 训练步演示
    decoder.train()
    for step in range(5):
        optimizer.zero_grad()
        loss = decoder(fused_features, target_reasoning_texts)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1} | Reasoning Loss: {loss.item():.4f}")

    # 2. 推理生成演示
    results = decoder.generate(
        fused_features,
        prompt_texts=["Reasoning: ", "Reasoning: "],
        max_new_tokens=24,
        do_sample=False,
    )
    for idx, text in enumerate(results):
        print(f"样本 {idx} 生成结果: {text}")