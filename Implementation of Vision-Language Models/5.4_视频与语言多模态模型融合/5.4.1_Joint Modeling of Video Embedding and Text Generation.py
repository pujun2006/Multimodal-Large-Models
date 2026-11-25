import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class VideoEmbeddingModel(nn.Module):
    """提取逐帧视觉特征并通过 LSTM 建模时序动态，输出视频级嵌入。"""

    def __init__(self, lstm_hidden_size: int = 512, lstm_layers: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        # 保留卷积特征提取层 + 全局平均池化（去掉全连接分类头 fc）
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.temporal_model = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [B, T, C, H, W]
        Returns:
            video_embedding: [B, lstm_hidden_size]
        """
        batch_size, num_frames, channels, height, width = video_frames.shape
        # 将时间维度合并进 batch 维做并行帧特征提取
        flat_frames = video_frames.view(batch_size * num_frames, channels, height, width)
        frame_feats = self.feature_extractor(flat_frames).flatten(1)  # [B*T, 2048]
        seq_feats = frame_feats.view(batch_size, num_frames, -1)  # [B, T, 2048]

        _, (hidden_state, _) = self.temporal_model(seq_feats)
        return hidden_state[-1]  # 取最后一层隐藏状态: [B, lstm_hidden_size]


class VideoToTextJointModel(nn.Module):
    """将视频嵌入投影至语言模型隐空间，并与文本 Token 联合进行自回归建模。"""

    def __init__(
        self,
        video_embedding_dim: int,
        model_name: str = "gpt2",
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
        llm_hidden_size = self.gpt_model.config.hidden_size

        if freeze_llm:
            for param in self.gpt_model.parameters():
                param.requires_grad = False

        # 两层 MLP 投影器，将视频嵌入映射为 1 个视觉前缀 Token
        self.embedding_projector = nn.Sequential(
            nn.Linear(video_embedding_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def _build_joint_inputs(
        self,
        video_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ):
        batch_size, _ = input_ids.shape
        device = input_ids.device

        # 1. 视频前缀嵌入: [B, 1, D]
        vis_embeds = self.embedding_projector(video_embedding).unsqueeze(1)
        # 2. 文本词嵌入: [B, L, D]
        text_embeds = self.gpt_model.transformer.wte(input_ids)
        # 3. 拼接为统一序列: [B, 1 + L, D]
        inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        vis_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        joint_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 4. 构造标签：视觉前缀位置设为 -100，不参与词元预测损失；有效文本位预测自身（HF 内部自动做 shift）
        ignore_prefix = torch.full((batch_size, 1), -100, device=device, dtype=torch.long)
        labels = torch.cat([ignore_prefix, input_ids.to(dtype=torch.long)], dim=1)
        labels = labels.masked_fill(joint_attention_mask == 0, -100)
        if pad_token_id is not None:
            labels = labels.masked_fill(labels == pad_token_id, -100)

        return inputs_embeds, joint_attention_mask, labels

    def forward(
        self,
        video_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ):
        inputs_embeds, joint_attention_mask, labels = self._build_joint_inputs(
            video_embedding=video_embedding,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )
        outputs = self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=joint_attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate_caption(
        self,
        video_embedding: torch.Tensor,
        tokenizer: GPT2Tokenizer,
        max_new_tokens: int = 20,
        prompt_text: str | None = None,
    ) -> list[str]:
        """基于视频嵌入执行自回归文本生成（支持批量推理）。"""
        self.eval()
        device = video_embedding.device
        batch_size = video_embedding.size(0)

        vis_embeds = self.embedding_projector(video_embedding).unsqueeze(1)  # [B, 1, D]

        if prompt_text:
            encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
            cur_ids = encoded["input_ids"].repeat(batch_size, 1)
            cur_embeds = torch.cat([vis_embeds, self.gpt_model.transformer.wte(cur_ids)], dim=1)
        else:
            cur_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            cur_embeds = vis_embeds

        eos_id = tokenizer.eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self.gpt_model(inputs_embeds=cur_embeds)
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [B]

            if eos_id is not None:
                next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_id), next_tokens)
                finished = finished | (next_tokens == eos_id)

            cur_ids = torch.cat([cur_ids, next_tokens.unsqueeze(1)], dim=1)
            next_embed = self.gpt_model.transformer.wte(next_tokens).unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

            if finished.all():
                break

        return [
            tokenizer.decode(seq.tolist(), skip_special_tokens=True).strip()
            for seq in cur_ids
        ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 1. 初始化 Tokenizer（GPT-2 默认无 pad_token，显式复用 eos_token）
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 构造示例数据：2 个视频，每个 16 帧，分辨率 224x224
    video_frames = torch.randn((2, 16, 3, 224, 224), device=device)
    # 图像归一化（匹配 ImageNet 预训练 ResNet50 分布）
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    video_frames = normalize(video_frames)

    target_texts = [
        "A person walking outdoors in sunny weather.",
        "A close-up shot of hands preparing food in a kitchen.",
    ]
    batch_enc = tokenizer(
        target_texts,
        max_length=24,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch_enc["input_ids"].to(device)
    attention_mask = batch_enc["attention_mask"].to(device)

    # 3. 初始化模型与优化器
    video_model = VideoEmbeddingModel(lstm_hidden_size=512, lstm_layers=2, pretrained=True).to(device)
    joint_model = VideoToTextJointModel(
        video_embedding_dim=512,
        model_name="gpt2",
        freeze_llm=False,  # 若先做特征对齐，可设为 True 仅训练 projector 与 temporal_model
    ).to(device)

    optimizer = optim.AdamW(
        list(video_model.parameters()) + list(joint_model.parameters()),
        lr=5e-5,
        weight_decay=1e-2,
    )

    # 4. 联合训练循环
    video_model.train()
    joint_model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        video_emb = video_model(video_frames)
        loss, _ = joint_model(
            video_embedding=video_emb,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(video_model.parameters()) + list(joint_model.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        print(f"Epoch {epoch + 1}/5 | Loss: {loss.item():.4f}")

    # 5. 推理阶段（单条测试视频）
    video_model.eval()
    joint_model.eval()
    test_video = normalize(torch.randn((1, 16, 3, 224, 224), device=device))
    with torch.no_grad():
        test_emb = video_model(test_video)
        captions = joint_model.generate_caption(
            video_embedding=test_emb,
            tokenizer=tokenizer,
            max_new_tokens=16,
        )
    print("生成的视频描述:", captions[0])


if __name__ == "__main__":
    main()