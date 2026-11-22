import torch
import torch.nn as nn


class ImageModalityEncoder(nn.Module):
    """简易视觉编码器：将图像划分为 Patch 并通过 Transformer 编码为特征序列"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.num_patches = (img_size // patch_size) ** 2

        # 使用步长等于 patch_size 的卷积完成 Patchify 与线性映射
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            tokens: [B, N_img, embed_dim]
        """
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, N_img, embed_dim]
        x = self.patch_proj(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)


class AudioModalityEncoder(nn.Module):
    """简易音频频谱编码器：对时间-频率特征做下采样与时序编码"""

    def __init__(
        self,
        n_mels: int = 80,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        max_audio_tokens: int = 256,
    ):
        super().__init__()
        # 局部时频特征提取与时间维度下采样（stride=2）
        self.conv_stem = nn.Sequential(
            nn.Conv1d(n_mels, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_audio_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [B, n_mels, T]
        Returns:
            tokens: [B, N_aud, embed_dim]
        """
        # [B, n_mels, T] -> [B, embed_dim, T'] -> [B, T', embed_dim]
        x = self.conv_stem(mel_spec).transpose(1, 2)
        seq_len = x.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError(
                f"音频序列长度 {seq_len} 超出最大位置编码长度 {self.pos_embed.size(1)}"
            )
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.encoder(x)
        return self.norm(x)


class ModalityProjector(nn.Module):
    """连接器（MLP 投影）：将不同编码器的特征映射到 LLM 统一的隐空间维度"""

    def __init__(self, in_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    llm_hidden_size = 768

    # 模拟输入：图像 [2, 3, 224, 224]，梅尔频谱 [2, 80, 400]
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_audio = torch.randn(batch_size, 80, 400)

    img_encoder = ImageModalityEncoder(img_size=224, patch_size=16, embed_dim=384)
    aud_encoder = AudioModalityEncoder(n_mels=80, embed_dim=384, max_audio_tokens=256)

    img_connector = ModalityProjector(in_dim=img_encoder.out_dim, llm_dim=llm_hidden_size)
    aud_connector = ModalityProjector(in_dim=aud_encoder.out_dim, llm_dim=llm_hidden_size)

    img_feats = img_encoder(dummy_images)          # [2, 196, 384]
    aud_feats = aud_encoder(dummy_audio)          # [2, 100, 384]

    img_llm_tokens = img_connector(img_feats)     # [2, 196, 768]
    aud_llm_tokens = aud_connector(aud_feats)     # [2, 100, 768]

    # 拼接为统一的多模态特征序列，后续可与文本 Embedding 拼接送入 LLM
    multimodal_tokens = torch.cat([img_llm_tokens, aud_llm_tokens], dim=1)

    print("图像编码输出形状:", img_feats.shape)
    print("音频编码输出形状:", aud_feats.shape)
    print("对齐后多模态序列形状:", multimodal_tokens.shape)  # [2, 296, 768]