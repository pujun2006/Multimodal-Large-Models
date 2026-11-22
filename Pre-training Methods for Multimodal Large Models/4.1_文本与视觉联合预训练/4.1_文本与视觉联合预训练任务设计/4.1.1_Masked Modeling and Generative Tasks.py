import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    GPT2LMHeadModel,
)

# 1. 基础配置与设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 加载双向编码器（用于多模态 MLM）与因果语言模型（用于视觉条件生成）
bert_model_name = "bert-base-uncased"
gpt_model_name = "gpt2"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_mlm = BertForMaskedLM.from_pretrained(bert_model_name).to(device)
bert_mlm.eval()

gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_lm = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
gpt_lm.eval()

# GPT-2 默认没有 pad_token，设置 pad_token_id 便于批处理与注意力掩码对齐
if gpt_tokenizer.pad_token_id is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_lm.config.pad_token_id = gpt_tokenizer.pad_token_id


class VisualPrefixProjector(nn.Module):
    """将模拟的视觉 Patch 序列 (B, N_vis, d_vis) 投影对齐到目标语言模型隐层维度 (B, N_vis, d_model)"""

    def __init__(self, d_vis: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vis, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_patches: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(visual_patches))


# ---------------------------------------------------------
# 2. 构造模拟视觉特征（Batch=1, 4个视觉 Patch, 每个维度 512）
# ---------------------------------------------------------
batch_size = 1
num_vis_tokens = 4
d_vis = 512
simulated_visual_patches = torch.randn(batch_size, num_vis_tokens, d_vis, device=device)

# =========================================================
# 3. 任务一：多模态掩码语言建模（Multimodal MLM）
# =========================================================
masked_text = "The quick brown fox jumps over the lazy [MASK]."
enc_inputs = bert_tokenizer(masked_text, return_tensors="pt").to(device)

bert_d_model = bert_mlm.config.hidden_size  # 768
vis_to_bert_proj = VisualPrefixProjector(d_vis=d_vis, d_model=bert_d_model).to(device)
vis_to_bert_proj.eval()

with torch.no_grad():
    # 获取文本词嵌入 (B, L_txt, d_model)
    txt_embeds_bert = bert_mlm.bert.embeddings.word_embeddings(enc_inputs["input_ids"])
    # 获取视觉前缀嵌入 (B, N_vis, d_model)
    vis_embeds_bert = vis_to_bert_proj(simulated_visual_patches)

    # 拼接为统一输入序列: [Visual Tokens, Text Tokens]
    # 注意：这里直接拼接嵌入向量；若需严格位置编码可再经 embeddings 层或自定义 position_ids
    fused_embeds_mlm = torch.cat([vis_embeds_bert, txt_embeds_bert], dim=1)

    # 构造对应的 attention_mask：视觉部分全可见（1），文本部分继承原有 mask
    vis_attn_mask = torch.ones(
        (batch_size, num_vis_tokens), dtype=enc_inputs["attention_mask"].dtype, device=device
    )
    fused_attn_mask_mlm = torch.cat([vis_attn_mask, enc_inputs["attention_mask"]], dim=1)

    # 送入 BERT 编码器与 MLM 预测头（跳过 word_embeddings 查表，直接传入 inputs_embeds）
    mlm_outputs = bert_mlm(
        inputs_embeds=fused_embeds_mlm,
        attention_mask=fused_attn_mask_mlm,
    )
    mlm_logits = mlm_outputs.logits  # 形状: (B, N_vis + L_txt, vocab_size)

    # 因为前面拼接了 num_vis_tokens 个视觉 Token，文本 [MASK] 的索引需要加上偏移量
    txt_input_ids_list = enc_inputs["input_ids"][0].tolist()
    txt_mask_pos = txt_input_ids_list.index(bert_tokenizer.mask_token_id)
    fused_mask_pos = num_vis_tokens + txt_mask_pos

    top_k = torch.topk(mlm_logits[0, fused_mask_pos], k=3)
    top_words = [bert_tokenizer.decode([idx]).strip() for idx in top_k.indices.tolist()]

    # 若有真实目标词，可构造 labels 计算交叉熵损失（非预测位置设为 -100）
    target_word_id = bert_tokenizer.encode("dog", add_special_tokens=False)[0]
    mlm_labels = torch.full(
        (batch_size, fused_embeds_mlm.size(1)), -100, dtype=torch.long, device=device
    )
    mlm_labels[0, fused_mask_pos] = target_word_id
    mlm_loss_outputs = bert_mlm(
        inputs_embeds=fused_embeds_mlm,
        attention_mask=fused_attn_mask_mlm,
        labels=mlm_labels,
    )

print(f"[多模态 MLM] [MASK] Top-3 预测词: {top_words}")
print(f"[多模态 MLM] 掩码位置交叉熵 Loss: {mlm_loss_outputs.loss.item():.4f}")

# =========================================================
# 4. 任务二：视觉条件自回归生成（Vision-Conditioned Generation）
# =========================================================
prompt_text = "Once upon a time"
gen_inputs = gpt_tokenizer(prompt_text, return_tensors="pt").to(device)

gpt_d_model = gpt_lm.config.n_embd  # 768
vis_to_gpt_proj = VisualPrefixProjector(d_vis=d_vis, d_model=gpt_d_model).to(device)
vis_to_gpt_proj.eval()

with torch.no_grad():
    # 获取文本 Token 嵌入 (B, L_prompt, d_model)
    prompt_embeds_gpt = gpt_lm.transformer.wte(gen_inputs["input_ids"])
    # 获取视觉前缀嵌入 (B, N_vis, d_model)
    vis_embeds_gpt = vis_to_gpt_proj(simulated_visual_patches)

    # 视觉前缀在前，文本提示在后，形成自回归上下文
    prefix_and_prompt_embeds = torch.cat([vis_embeds_gpt, prompt_embeds_gpt], dim=1)
    prefix_and_prompt_mask = torch.cat(
        [
            torch.ones((batch_size, num_vis_tokens), dtype=gen_inputs["attention_mask"].dtype, device=device),
            gen_inputs["attention_mask"],
        ],
        dim=1,
    )

    # 4a. 训练/评估视角：计算下一个词元预测损失（仅对文本部分计算监督损失，视觉前缀设为 -100）
    target_text = "Once upon a time there was a curious fox."
    target_ids = gpt_tokenizer(target_text, return_tensors="pt")["input_ids"].to(device)
    full_embeds = torch.cat(
        [vis_embeds_gpt, gpt_lm.transformer.wte(target_ids)], dim=1
    )
    full_mask = torch.cat(
        [
            torch.ones((batch_size, num_vis_tokens), dtype=torch.long, device=device),
            torch.ones_like(target_ids),
        ],
        dim=1,
    )
    gen_labels = torch.cat(
        [
            torch.full((batch_size, num_vis_tokens), -100, dtype=torch.long, device=device),
            target_ids,
        ],
        dim=1,
    )
    gen_loss_outputs = gpt_lm(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        labels=gen_labels,
    )

    # 4b. 推理视角：以视觉前缀 + 文本提示嵌入为条件进行自回归续写
    # 注意：使用 inputs_embeds 时，max_new_tokens 控制新生成的词元数更直观
    generated_ids = gpt_lm.generate(
        inputs_embeds=prefix_and_prompt_embeds,
        attention_mask=prefix_and_prompt_mask,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=gpt_tokenizer.pad_token_id,
    )
    generated_continuation = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"[视觉条件生成] 文本目标交叉熵 Loss: {gen_loss_outputs.loss.item():.4f}")
print(f"[视觉条件生成] 续写输出（基于视觉前缀 + Prompt）: {generated_continuation}")