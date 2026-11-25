import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, processor, tokenizer, max_length=32):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ImageCaptioningModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", language_model_name="gpt2"):
        super().__init__()
        self.vision_model = CLIPModel.from_pretrained(clip_model_name).vision_model
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        self.fc = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size,
        )

    def encode_image(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        return self.fc(vision_outputs.pooler_output).unsqueeze(1)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        batch_size = pixel_values.size(0)
        img_embeds = self.encode_image(pixel_values)
        tok_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([img_embeds, tok_embeds], dim=1)

        img_attn = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attn = torch.cat([img_attn, attention_mask], dim=1)

        img_labels = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([img_labels, labels], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate_caption(self, pixel_values, max_new_tokens=20):
        self.eval()
        img_embeds = self.encode_image(pixel_values)  # [B, 1, D]
        generated_ids = self.language_model.generate(
            inputs_embeds=img_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    # 替换为你的真实图片路径和对应描述
    image_paths = ["image1.jpg", "image2.jpg"]
    captions = ["A wet, folded black umbrella.", "A glass of water sits on a rustic wooden table in a cozy room."]

    dataset = ImageCaptionDataset(image_paths, captions, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ImageCaptioningModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # 1. 打印前向传播测试结果
    batch = next(iter(dataloader))
    loss, logits = model(
        batch["pixel_values"].to(device),
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )
    print(f"初始 Loss: {loss.item():.4f}, Logits 形状: {logits.shape}")

    # 2. 简单训练示例（演示梯度更新）
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for b in dataloader:
            optimizer.zero_grad()
            l, _ = model(
                b["pixel_values"].to(device),
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["labels"].to(device),
            )
            l.backward()
            optimizer.step()
            total_loss += l.item()
        print(f"Epoch {epoch + 1} 平均 Loss: {total_loss / len(dataloader):.4f}")

    # 3. 打印推理生成结果
    sample_pixels = batch["pixel_values"].to(device)
    preds = model.generate_caption(sample_pixels)
    for i, text in enumerate(preds):
        print(f"样本 {i} 生成描述: {text}")