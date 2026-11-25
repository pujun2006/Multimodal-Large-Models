import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class TrainConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    gpt_model_name: str = "gpt2"
    prompt: str = "A photo of "
    num_prefix_tokens: int = 8
    max_length: int = 64
    batch_size: int = 4
    stage1_epochs: int = 10
    stage2_epochs: int = 6
    projector_lr: float = 2e-4
    lora_lr: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    val_ratio: float = 0.2
    save_path: str = "best_multimodal_captioner.pt"
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VisualPrefixProjector(nn.Module):
    """将归一化后的 CLIP 视觉特征映射为多个 Prefix Token，并带可学习尺度缩放"""

    def __init__(self, vis_dim: int, txt_dim: int, num_prefix_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.txt_dim = txt_dim
        hidden_dim = max(txt_dim, vis_dim * 2)

        self.net = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_prefix_tokens * txt_dim),
            nn.LayerNorm(num_prefix_tokens * txt_dim),
        )
        # 初始缩放因子设为 0.5，帮助视觉前缀平稳融入文本嵌入分布
        self.visual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: [B, vis_dim]
        normed = F.normalize(image_features.float(), dim=-1)
        projected = self.net(normed)
        bsz = projected.size(0)
        tokens = projected.view(bsz, self.num_prefix_tokens, self.txt_dim)
        return tokens * self.visual_scale


class MultimodalCaptionGenerator(nn.Module):
    """CLIP 视觉编码器 + 视觉前缀投影器 + GPT-2 解码器"""

    def __init__(
        self,
        clip_model: CLIPModel,
        text_decoder: nn.Module,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.text_decoder = text_decoder
        self.num_prefix_tokens = num_prefix_tokens

        vis_dim = clip_model.config.projection_dim
        txt_dim = text_decoder.config.n_embd
        self.projector = VisualPrefixProjector(
            vis_dim=vis_dim,
            txt_dim=txt_dim,
            num_prefix_tokens=num_prefix_tokens,
        )

    def _get_word_embeddings(self) -> nn.Module:
        # 兼容 PeftModel 包装后的 GPT2LMHeadModel
        base = getattr(self.text_decoder, "base_model", self.text_decoder)
        if hasattr(base, "model") and hasattr(base.model, "transformer"):
            return base.model.transformer.wte
        if hasattr(base, "transformer"):
            return base.transformer.wte
        return self.text_decoder.get_input_embeddings()

    def encode_visual_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vis_out = self.clip_model.get_image_features(pixel_values=pixel_values)
            if not isinstance(vis_out, torch.Tensor):
                vis_out = vis_out.pooler_output
        return self.projector(vis_out)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        device = pixel_values.device
        bsz, seq_len = input_ids.shape

        visual_embeds = self.encode_visual_prefix(pixel_values).to(dtype=self._get_word_embeddings().weight.dtype)
        text_embeds = self._get_word_embeddings()(input_ids)

        # [B, P + T, D]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=device)
        prefix_mask = torch.ones((bsz, self.num_prefix_tokens), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )
        # 仅取文本部分对应的 logits: [B, T, V]
        text_logits = outputs.logits[:, self.num_prefix_tokens :, :]

        loss = None
        if labels is not None:
            # 标准因果语言建模错位：用 t 位置的 logits 预测 t+1 位置的 label
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        return loss, text_logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        prompt: str = "A photo of ",
        max_new_tokens: int = 25,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        self.eval()
        device = pixel_values.device

        visual_embeds = self.encode_visual_prefix(pixel_values).to(dtype=self._get_word_embeddings().weight.dtype)
        bsz = visual_embeds.size(0)

        prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = self._get_word_embeddings()(prompt_enc["input_ids"])

        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        prefix_mask = torch.ones((bsz, self.num_prefix_tokens), dtype=prompt_enc["attention_mask"].dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, prompt_enc["attention_mask"]], dim=1)

        gen_ids = self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_suffix = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return f"{prompt}{generated_suffix}".strip()


class CaptionDataset(Dataset):
    """图文对数据集：统一构造 input_ids、attention_mask 与精确目标 labels"""

    def __init__(
        self,
        samples: List[Dict[str, str]],
        clip_processor: CLIPProcessor,
        tokenizer,
        prompt: str = "A photo of ",
        max_length: int = 64,
    ):
        self.samples = samples
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        full_text = f"{self.prompt}{item['caption'].strip()}{self.tokenizer.eos_token}"
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # 屏蔽 prompt 前缀的损失，专注于学习图像描述内容
        prompt_ids = self.tokenizer(self.prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), labels.size(0))
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def configure_trainable_params(model: MultimodalCaptionGenerator, stage: int) -> None:
    """
    stage=1: 冻结 CLIP 与 GPT-2，仅训练 projector（特征对齐）
    stage=2: 保持 CLIP 冻结，训练 projector + LoRA（若安装了 peft）或解冻 GPT-2 顶层块
    """
    for p in model.clip_model.parameters():
        p.requires_grad = False

    if stage == 1:
        for p in model.text_decoder.parameters():
            p.requires_grad = False
        for p in model.projector.parameters():
            p.requires_grad = True
    elif stage == 2:
        for p in model.projector.parameters():
            p.requires_grad = True
        if HAS_PEFT and hasattr(model.text_decoder, "peft_config"):
            for name, p in model.text_decoder.named_parameters():
                p.requires_grad = "lora_" in name
        else:
            # 无 peft 时的兜底：仅微调最后 2 个 Transformer block 与最终 LayerNorm
            for p in model.text_decoder.parameters():
                p.requires_grad = False
            base = getattr(model.text_decoder, "base_model", model.text_decoder)
            if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
                for block in base.transformer.h[-2:]:
                    for p in block.parameters():
                        p.requires_grad = True
                if hasattr(base.transformer, "ln_f"):
                    for p in base.transformer.ln_f.parameters():
                        p.requires_grad = True


def build_optimizer_and_scheduler(
    model: MultimodalCaptionGenerator,
    total_steps: int,
    projector_lr: float,
    lora_lr: float,
    weight_decay: float,
):
    param_groups = []
    proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    if proj_params:
        param_groups.append({"params": proj_params, "lr": projector_lr, "weight_decay": weight_decay})

    dec_params = [p for p in model.text_decoder.parameters() if p.requires_grad]
    if dec_params:
        param_groups.append({"params": dec_params, "lr": lora_lr, "weight_decay": weight_decay})

    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(
    model: MultimodalCaptionGenerator,
    val_loader: Optional[DataLoader],
    device: torch.device,
    label_smoothing: float = 0.0,
) -> Optional[float]:
    if val_loader is None or len(val_loader) == 0:
        return None
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        loss, _ = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            label_smoothing=label_smoothing,
        )
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def run_training_stage(
    model: MultimodalCaptionGenerator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    stage: int,
    cfg: TrainConfig,
    best_val_loss: float,
) -> float:
    configure_trainable_params(model, stage=stage)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage {stage}] 可训练参数量: {trainable_count:,}")

    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        total_steps=total_steps,
        projector_lr=cfg.projector_lr,
        lora_lr=cfg.lora_lr if stage == 2 else 0.0,
        weight_decay=cfg.weight_decay,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
                loss, _ = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=cfg.label_smoothing,
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.grad_clip,
                )
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device, label_smoothing=0.0)

        if val_loss is not None:
            print(f"Stage {stage} | Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.__dict__}, cfg.save_path)
        else:
            print(f"Stage {stage} | Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.__dict__}, cfg.save_path)

    return best_val_loss


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model = CLIPModel.from_pretrained(cfg.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    text_decoder: nn.Module = GPT2LMHeadModel.from_pretrained(cfg.gpt_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        text_decoder.config.pad_token_id = tokenizer.eos_token_id

    if HAS_PEFT:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
        )
        text_decoder = get_peft_model(text_decoder, lora_cfg)

    text_decoder = text_decoder.to(device)
    model = MultimodalCaptionGenerator(
        clip_model=clip_model,
        text_decoder=text_decoder,
        num_prefix_tokens=cfg.num_prefix_tokens,
    ).to(device)

    # 示例数据（实际使用时请替换为自己的图文对列表）
    raw_samples = [
        {"image_path": "cat.jpg", "caption": "a tabby cat with green eyes sits on a light-colored ledge"},
        {"image_path": "dog.jpg", "caption": "a yellow dog stands on a sandy beach"},
        {"image_path": "bird.jpg", "caption": "a colorful bird perched on a branch"},
        {"image_path": "cat.jpg", "caption": "a close-up photo of a cat"},
    ]
    valid_samples = [s for s in raw_samples if os.path.exists(s["image_path"])]

    if len(valid_samples) >= 2:
        full_ds = CaptionDataset(
            samples=valid_samples,
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            prompt=cfg.prompt,
            max_length=cfg.max_length,
        )
        val_size = max(1, int(len(full_ds) * cfg.val_ratio)) if len(full_ds) >= 5 else 0
        train_size = len(full_ds) - val_size

        if val_size > 0:
            train_ds, val_ds = random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = None

        best_val_loss = math.inf
        best_val_loss = run_training_stage(
            model, train_loader, val_loader, device, cfg.stage1_epochs, stage=1, cfg=cfg, best_val_loss=best_val_loss
        )
        best_val_loss = run_training_stage(
            model, train_loader, val_loader, device, cfg.stage2_epochs, stage=2, cfg=cfg, best_val_loss=best_val_loss
        )

        if os.path.exists(cfg.save_path):
            ckpt = torch.load(cfg.save_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

    # 推理测试（无本地图片时自动用随机图像张量验证完整前向与生成通路）
    if os.path.exists("cat.jpg"):
        img = Image.open("cat.jpg").convert("RGB")
        pixel_values = clip_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    else:
        print("未检测到本地图片，使用随机张量验证推理通路...")
        pixel_values = torch.randn((1, 3, 224, 224), device=device)

    caption = model.generate(
        pixel_values=pixel_values,
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        max_new_tokens=25,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    print("生成结果:", caption)


if __name__ == "__main__":
    main()