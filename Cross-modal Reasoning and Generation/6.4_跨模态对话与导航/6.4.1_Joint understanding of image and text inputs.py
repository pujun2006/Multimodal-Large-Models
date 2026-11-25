import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPModel,
    CLIPProcessor,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from PIL import Image, ImageDraw


class MultimodalDialogueModel(nn.Module):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        gpt_name: str = "gpt2",
        num_visual_tokens: int = 4,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_name)

        clip_dim = self.clip_model.config.projection_dim  # 512 for vit-base-patch32
        gpt_dim = self.gpt_model.config.n_embd            # 768 for gpt2
        self.num_visual_tokens = num_visual_tokens

        # 将全局图像特征映射为 num_visual_tokens 个视觉 token 表示，送入 GPT-2 序列
        self.image_proj = nn.Sequential(
            nn.Linear(clip_dim, gpt_dim * num_visual_tokens),
            nn.GELU(),
            nn.LayerNorm(gpt_dim * num_visual_tokens),
        )

    def encode_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W] -> image_embeds: [B, clip_dim]
        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.pooler_output
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)

        # 映射到 [B, num_visual_tokens, gpt_dim]
        projected = self.image_proj(image_embeds)
        visual_tokens = projected.view(
            projected.size(0), self.num_visual_tokens, -1
        )
        return visual_tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ):
        visual_embeds = self.encode_visual_tokens(pixel_values)  # [B, V, D]
        batch_size, _, gpt_dim = visual_embeds.shape
        device = visual_embeds.device

        wte = self.gpt_model.transformer.wte
        prompt_embeds = wte(prompt_input_ids)  # [B, T_p, D]

        visual_mask = torch.ones(
            (batch_size, self.num_visual_tokens),
            dtype=prompt_attention_mask.dtype,
            device=device,
        )

        if target_input_ids is not None:
            # 训练模式：将 [视觉 tokens + 提问 tokens + 回复 tokens] 拼接
            target_embeds = wte(target_input_ids)  # [B, T_r, D]
            inputs_embeds = torch.cat(
                [visual_embeds, prompt_embeds, target_embeds], dim=1
            )

            if target_attention_mask is None:
                target_attention_mask = torch.ones_like(target_input_ids)

            attention_mask = torch.cat(
                [visual_mask, prompt_attention_mask, target_attention_mask], dim=1
            )

            # 仅对 target_input_ids 计算损失，视觉与提问部分置为 -100
            prefix_len = self.num_visual_tokens + prompt_input_ids.size(1)
            ignore_prefix = torch.full(
                (batch_size, prefix_len),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat([ignore_prefix, target_input_ids], dim=1)

            # 将 target_attention_mask 对齐到与 labels 相同的总长度 [B, V + T_p + T_r]
            full_target_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, prefix_len),
                        dtype=target_attention_mask.dtype,
                        device=device,
                    ),
                    target_attention_mask,
                ],
                dim=1,
            )
            labels = labels.masked_fill(full_target_mask == 0, -100)

            return self.gpt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # 仅前向推理 logits 模式
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([visual_mask, prompt_attention_mask], dim=1)
        return self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate_reply(
        self,
        pixel_values: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int = 30,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        self.eval()
        visual_embeds = self.encode_visual_tokens(pixel_values)  # [B, V, D]
        prompt_embeds = self.gpt_model.transformer.wte(prompt_input_ids)

        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        visual_mask = torch.ones(
            (pixel_values.size(0), self.num_visual_tokens),
            dtype=prompt_attention_mask.dtype,
            device=pixel_values.device,
        )
        attention_mask = torch.cat([visual_mask, prompt_attention_mask], dim=1)

        generated_ids = self.gpt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.gpt_model.config.pad_token_id,
            eos_token_id=self.gpt_model.config.eos_token_id,
        )
        return generated_ids


class MultimodalQADataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        clip_processor: CLIPProcessor,
        tokenizer: GPT2Tokenizer,
    ):
        self.samples = samples
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        prompt_text = f"Question: {item['question']} Answer:"
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        out = {
            "pixel_values": pixel_values,
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
        }

        if "answer" in item and item["answer"] is not None:
            answer_text = " " + item["answer"] + (self.tokenizer.eos_token or "")
            target_enc = self.tokenizer(
                answer_text,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out["target_input_ids"] = target_enc["input_ids"].squeeze(0)
            out["target_attention_mask"] = target_enc["attention_mask"].squeeze(0)

        return out


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[dict]) -> dict:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

        max_p = max(b["prompt_input_ids"].size(0) for b in batch)
        prompt_input_ids = torch.full((len(batch), max_p), pad_token_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((len(batch), max_p), dtype=torch.long)

        for i, b in enumerate(batch):
            l = b["prompt_input_ids"].size(0)
            prompt_input_ids[i, :l] = b["prompt_input_ids"]
            prompt_attention_mask[i, :l] = b["prompt_attention_mask"]

        result = {
            "pixel_values": pixel_values,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }

        if "target_input_ids" in batch[0]:
            max_t = max(b["target_input_ids"].size(0) for b in batch)
            target_input_ids = torch.full((len(batch), max_t), pad_token_id, dtype=torch.long)
            target_attention_mask = torch.zeros((len(batch), max_t), dtype=torch.long)
            for i, b in enumerate(batch):
                l = b["target_input_ids"].size(0)
                target_input_ids[i, :l] = b["target_input_ids"]
                target_attention_mask[i, :l] = b["target_attention_mask"]
            result["target_input_ids"] = target_input_ids
            result["target_attention_mask"] = target_attention_mask

        return result

    return collate_fn


def train_model(
    model: MultimodalDialogueModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            prompt_ids = batch["prompt_input_ids"].to(device)
            prompt_mask = batch["prompt_attention_mask"].to(device)
            target_ids = batch["target_input_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                target_input_ids=target_ids,
                target_attention_mask=target_mask,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / max(1, len(dataloader)):.4f}")


def evaluate_model(
    model: MultimodalDialogueModel,
    dataloader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
):
    model.eval()
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)

        generated_ids = model.generate_reply(
            pixel_values=pixel_values,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            max_new_tokens=25,
        )
        for i in range(generated_ids.size(0)):
            prompt_str = tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            gen_str = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            print(f"Prompt: {prompt_str}")
            print(f"Generated: {gen_str}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 生成示例图像
    img_path = "example_image.jpg"
    img = Image.new("RGB", (224, 224), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 184, 184], outline=(30, 30, 30), width=4)
    draw.text((75, 105), "CAT", fill=(10, 10, 10))
    img.save(img_path)

    # 2. 初始化处理器、分词器与模型
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MultimodalDialogueModel(num_visual_tokens=4).to(device)
    model.gpt_model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 构造示例训练与测试数据（如需测试真实图片，可将 image_path 替换为实际文件路径）
    train_samples = [
        {"image_path": "image1.jpg", "question": "What is shown in the image?", "answer": "A wet, folded black umbrella."},
        {"image_path": "image2.jpg", "question": "Describe the picture briefly.", "answer": "A glass of water sits on a rustic wooden table in a cozy room."},
    ]
    test_samples = [
        {"image_path": "test_image.jpg", "question": "What is shown in the image?"},
        {"image_path": "example_image.jpg", "question": "What is shown in the image?"},
    ]

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        MultimodalQADataset(train_samples, clip_processor, tokenizer),
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        MultimodalQADataset(test_samples, clip_processor, tokenizer),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 4. 冻结 CLIP 与 GPT-2 主干、仅训练映射层
    for p in model.clip_model.parameters():
        p.requires_grad = False
    for p in model.gpt_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
    )

    train_model(model, train_loader, optimizer, device=device, epochs=3)
    evaluate_model(model, test_loader, tokenizer, device=device)






