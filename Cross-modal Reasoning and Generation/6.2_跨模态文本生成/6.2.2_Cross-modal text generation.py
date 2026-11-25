import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, GPT2LMHeadModel
from PIL import Image


class MultimodalDataset(Dataset):
    def __init__(self, image_paths, captions, processor, tokenizer, max_length=50):
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
        text_enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
        }


class MultimodalTextGenerator(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        gpt_model_name="uer/gpt2-chinese-cluecorpussmall",
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        # 冻结 CLIP 视觉编码器
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vision_dim = self.clip_model.config.vision_config.hidden_size
        gpt_dim = self.gpt_model.config.n_embd
        self.fc = nn.Linear(vision_dim, gpt_dim)

    def encode_image(self, pixel_values):
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output  # (B, vision_dim)
        return self.fc(pooled_output).unsqueeze(1)    # (B, 1, gpt_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_embeds = self.encode_image(pixel_values)                 # (B, 1, D)
        txt_embeds = self.gpt_model.transformer.wte(input_ids)       # (B, L, D)
        inputs_embeds = torch.cat([img_embeds, txt_embeds], dim=1)   # (B, 1+L, D)

        img_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([img_mask, attention_mask], dim=1)

        ignore_img = torch.full(
            (input_ids.size(0), 1),
            -100,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        full_labels = torch.cat([ignore_img, labels], dim=1)

        outputs = self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return outputs.loss, outputs.logits


def generate_text(model, image_path, processor, tokenizer, max_new_tokens=30, device="cpu"):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        img_embeds = model.encode_image(pixel_values)  # (1, 1, D)

        # 若希望以 [CLS] 作为文本起始引导，可拼上 cls_token 的词嵌入
        if tokenizer.cls_token_id is not None:
            cls_id = torch.tensor([[tokenizer.cls_token_id]], device=device)
            cls_embed = model.gpt_model.transformer.wte(cls_id)
            inputs_embeds = torch.cat([img_embeds, cls_embed], dim=1)
        else:
            inputs_embeds = img_embeds

        generated_ids = model.gpt_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 调试期先用确定性解码，便于观察真实学到了什么
            num_beams=3,
            repetition_penalty=1.2,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    gpt_name = "uer/gpt2-chinese-cluecorpussmall"

    processor = CLIPProcessor.from_pretrained(clip_name)
    tokenizer = AutoTokenizer.from_pretrained(gpt_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    image_paths = ["image1.jpg", "image2.jpg"]
    captions = ["一把伞斜靠在门上。", "一个杯子放在桌子上。"]

    dataset = MultimodalDataset(image_paths, captions, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = MultimodalTextGenerator(clip_model_name=clip_name, gpt_model_name=gpt_name).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            loss, _ = model(pixel_values, input_ids, attention_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/5, Loss: {total_loss / len(dataloader):.4f}")

    print("Generated Text:", generate_text(model, "image1.jpg", processor, tokenizer, device=device))