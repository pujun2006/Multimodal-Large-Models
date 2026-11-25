import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# 1. 视频嵌入模型（轻量化视觉特征 + 时序 Transformer 编码）
class OptimizedVideoEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=1280):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.feature_extractor = models.mobilenet_v2(weights=weights).features
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # 可选：可学习的时间位置编码，帮助区分帧先后顺序
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, embed_dim))

    def forward(self, video_frames):
        # video_frames: (B, T, C, H, W)
        batch_size, num_frames, c, h, w = video_frames.size()
        flat_frames = video_frames.view(batch_size * num_frames, c, h, w)

        # 批量提取空间特征并做空间平均池化 -> (B * T, 1280)
        frame_features = self.feature_extractor(flat_frames).mean(dim=[2, 3])
        video_features = frame_features.view(batch_size, num_frames, self.embed_dim)

        if num_frames <= self.pos_embedding.size(1):
            video_features = video_features + self.pos_embedding[:, :num_frames, :]

        temporal_features = self.temporal_model(video_features)  # (B, T, 1280)
        # 返回时间池化特征 (B, 1280)
        return temporal_features.mean(dim=1)


# 2. 多模态文本生成模型（Visual Prefix + GPT-2）
class OptimizedVideoToTextModel(nn.Module):
    def __init__(self, video_embedding_dim, prefix_len=1):
        super().__init__()
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        hidden_size = self.gpt_model.config.hidden_size
        self.prefix_len = prefix_len

        # 将视频嵌入映射到 prefix_len * hidden_size，支持单前缀或多前缀 token
        self.embedding_projector = nn.Linear(
            video_embedding_dim, prefix_len * hidden_size
        )

    def build_prefix_embeds(self, video_embedding):
        # video_embedding: (B, video_embedding_dim) -> (B, prefix_len, hidden_size)
        batch_size = video_embedding.size(0)
        hidden_size = self.gpt_model.config.hidden_size
        projected = self.embedding_projector(video_embedding)
        return projected.view(batch_size, self.prefix_len, hidden_size)

    def forward(self, video_embedding, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        prefix_embeds = self.build_prefix_embeds(video_embedding)  # (B, P, H)
        token_embeds = self.gpt_model.transformer.wte(input_ids)    # (B, L, H)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (B, P+L, H)

        prefix_mask = torch.ones(
            (batch_size, self.prefix_len),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )
        # 只保留文本 token 对应的 logits，形状为 (B, L, vocab_size)
        text_logits = outputs.logits[:, self.prefix_len :, :]
        return text_logits

    @torch.no_grad()
    def generate_caption(
        self,
        video_embedding,
        tokenizer,
        max_new_tokens=20,
        prompt_ids=None,
    ):
        self.eval()
        device = video_embedding.device
        prefix_embeds = self.build_prefix_embeds(video_embedding)  # (B, P, H)

        if prompt_ids is not None:
            prompt_embeds = self.gpt_model.transformer.wte(prompt_ids)
            inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        else:
            inputs_embeds = prefix_embeds

        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        generated_ids = self.gpt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理（包含 ImageNet 归一化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 示例数据
    raw_videos = torch.rand((4, 8, 3, 224, 224))
    video_frames = torch.stack([
        torch.stack([transform(transforms.ToPILImage()(frame)) for frame in vid])
        for vid in raw_videos
    ]).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_texts = [
        "A person walking outdoors in daylight.",
        "A car driving down a city street.",
        "Someone cooking food in a kitchen.",
        "Waves crashing onto a rocky shore.",
    ]
    encoded = tokenizer(
        input_texts,
        max_length=20,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 初始化模型与优化器
    video_model = OptimizedVideoEmbeddingModel(embed_dim=1280).to(device)
    text_model = OptimizedVideoToTextModel(video_embedding_dim=1280, prefix_len=1).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(
        [
            {"params": video_model.parameters(), "lr": 1e-4},
            {"params": text_model.parameters(), "lr": 5e-5},
        ],
        weight_decay=1e-2,
    )

    # 训练阶段（错位计算 Next-Token Prediction Loss）
    for epoch in range(5):
        video_model.train()
        text_model.train()
        optimizer.zero_grad()

        video_embedding = video_model(video_frames)
        logits = text_model(video_embedding, input_ids, attention_mask=attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # 推理阶段（自回归生成完整描述）
    video_model.eval()
    text_model.eval()
    test_video = torch.rand((1, 8, 3, 224, 224), device=device)

    with torch.no_grad():
        test_embedding = video_model(test_video)
        generated_text = text_model.generate_caption(
            test_embedding, tokenizer, max_new_tokens=15
        )
        print("生成的文本描述:", generated_text)


if __name__ == "__main__":
    main()