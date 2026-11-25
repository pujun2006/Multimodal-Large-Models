import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
)


class ImageToTextGenerator(nn.Module):
    """基于 CLIP 视觉编码器 + MLP 投影层 + GPT-2 解码器的图像到文本模型"""

    def __init__(
        self,
        clip_model: CLIPModel,
        text_decoder: GPT2LMHeadModel,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.text_decoder = text_decoder
        self.num_prefix_tokens = num_prefix_tokens

        vis_dim = clip_model.config.projection_dim
        txt_dim = text_decoder.config.n_embd

        # 将视觉特征映射到文本嵌入空间，并扩展为 num_prefix_tokens 个软提示（Prefix Tokens）
        self.projector = nn.Sequential(
            nn.Linear(vis_dim, txt_dim * num_prefix_tokens),
            nn.GELU(),
            nn.Linear(txt_dim * num_prefix_tokens, txt_dim * num_prefix_tokens),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取归一化后的图像特征，并投影为 [B, num_prefix_tokens, txt_dim]"""
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        # 兼容部分 transformers 版本可能返回 BaseModelOutputWithPooling 的情况
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        projected = self.projector(image_features)

        batch_size = projected.size(0)
        txt_dim = self.text_decoder.config.n_embd
        return projected.view(batch_size, self.num_prefix_tokens, txt_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        训练/评估前向传播：
        - pixel_values: [B, 3, H, W]
        - input_ids: [B, T]
        - attention_mask: [B, T]
        - labels: [B, T]，非目标预测位置应设为 -100
        """
        device = pixel_values.device
        batch_size, seq_len = input_ids.shape

        # 1. 视觉前缀嵌入: [B, P, D]
        visual_embeds = self.encode_image(pixel_values)

        # 2. 文本词嵌入: [B, T, D]
        text_embeds = self.text_decoder.transformer.wte(input_ids)

        # 3. 沿序列维度拼接: [B, P + T, D]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 4. 拼接注意力掩码: [B, P + T]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        vis_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 5. 拼接标签（视觉前缀位置不参与语言模型 loss 计算，填 -100）
        full_labels = None
        if labels is not None:
            vis_labels = torch.full(
                (batch_size, self.num_prefix_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=device,
            )
            full_labels = torch.cat([vis_labels, labels], dim=1)

        return self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        prompt: str = "A photo of",
        max_new_tokens: int = 30,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """自回归生成图像描述"""
        self.eval()
        device = pixel_values.device

        # 1. 视觉前缀嵌入: [1, P, D]
        visual_embeds = self.encode_image(pixel_values)
        batch_size, _, _ = visual_embeds.shape

        # 2. Prompt 文本嵌入: [1, T_prompt, D]
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = self.text_decoder.transformer.wte(prompt_inputs["input_ids"])

        # 3. 拼接输入嵌入与注意力掩码
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        vis_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=prompt_inputs["attention_mask"].dtype,
            device=device,
        )
        full_attention_mask = torch.cat([vis_mask, prompt_inputs["attention_mask"]], dim=1)

        # 4. 自回归生成新 Token
        # 注意：传入 inputs_embeds 时，generate 返回的序列仅包含新生成的 token IDs
        generated_ids = self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=1.2,  # 抑制重复循环生成
            no_repeat_ngram_size=3,  # 禁止 3-gram 重复
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return f"{prompt}{generated_text}".strip()


class CaptionDataset(Dataset):
    """简单的图文对数据集示例"""

    def __init__(
        self,
        samples: List[dict],  # 每项形如 {"image_path": "...", "caption": "..."}
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

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        full_text = f"{self.prompt}{item['caption']}{self.tokenizer.eos_token}"
        prompt_text = f"{self.prompt}"

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # 构造 labels：padding 位置设为 -100；可选将 prompt 部分设为 -100 以专注预测描述内容
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        labels[: min(prompt_len, len(labels))] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_projector(
    model: ImageToTextGenerator,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
):
    """阶段一对齐训练：冻结 CLIP 与 GPT-2，仅更新 projector"""
    for param in model.clip_model.parameters():
        param.requires_grad = False
    for param in model.text_decoder.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True

    model.to(device)
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch}/{epochs}] - Avg Loss: {avg_loss:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model_name = "openai/clip-vit-base-patch32"
    gpt_model_name = "gpt2"

    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    text_decoder = GPT2LMHeadModel.from_pretrained(gpt_model_name)
    tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)

    # GPT-2 默认没有 pad_token，常将其设为 eos_token 以便批量 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        text_decoder.config.pad_token_id = tokenizer.eos_token_id

    model = ImageToTextGenerator(
        clip_model=clip_model,
        text_decoder=text_decoder,
        num_prefix_tokens=8,
    ).to(device)

    # 示例训练数据（请替换为真实本地图像路径与描述）
    sample_data = [
        {"image_path": "cat.jpg", "caption": "a tabby cat with green eyes sits on a light-colored ledge"},
        {"image_path": "dog.jpg", "caption": "a yellow dog stands on a sandy beach"},
        {"image_path": "bird.jpg", "caption": "a colorful bird perched on a branch"},
        {"image_path": "cat.jpg", "caption": "a close-up photo of a cat"},
    ]
    valid_samples = [s for s in sample_data if os.path.exists(s["image_path"])]

    if valid_samples:
        dataset = CaptionDataset(valid_samples, clip_processor, tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        train_projector(model, dataloader, device=device, epochs=80, lr=1e-4)

    # 推理测试（若没有本地图片，可先用随机张量验证代码通路是否畅通）
    if os.path.exists("cat.jpg"):
        image = Image.open("cat.jpg").convert("RGB")
        pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    else:
        print("未找到 cat.jpg，使用随机图像张量验证前向与生成通路...")
        pixel_values = torch.randn((1, 3, 224, 224), device=device)

    caption = model.generate_caption(
        pixel_values=pixel_values,
        tokenizer=tokenizer,
        prompt="A photo of ",
        max_new_tokens=20,
    )
    print("生成的描述文本:", caption)


if __name__ == "__main__":
    main()