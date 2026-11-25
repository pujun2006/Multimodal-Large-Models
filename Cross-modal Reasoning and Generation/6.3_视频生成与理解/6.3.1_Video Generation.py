import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer


# 1. 构造带简单时空规律的合成数据集（便于验证模型是否真的在学习）
def make_synthetic_videos(num_per_class=8, num_frames=10, size=64):
  captions = []
  sequences = []

  # 模式 A：红色方块从左向右移动
  for _ in range(num_per_class):
    frames = np.zeros((num_frames, size, size, 3), dtype=np.float32)
    start_x = np.random.randint(4, 16)
    y = np.random.randint(16, size - 24)
    for t in range(num_frames):
      x = min(size - 12, start_x + t * 3)
      frames[t, y : y + 10, x : x + 10, 0] = 1.0  # R 通道
    captions.append("一个红色方块从左向右移动。")
    sequences.append(frames)

  # 模式 B：绿色方块从上向下移动
  for _ in range(num_per_class):
    frames = np.zeros((num_frames, size, size, 3), dtype=np.float32)
    x = np.random.randint(16, size - 24)
    start_y = np.random.randint(4, 16)
    for t in range(num_frames):
      y = min(size - 12, start_y + t * 3)
      frames[t, y : y + 10, x : x + 10, 1] = 1.0  # G 通道
    captions.append("一个绿色方块从上向下移动。")
    sequences.append(frames)

  return captions, sequences


class VideoDataset(Dataset):

  def __init__(self, captions, frame_sequences, tokenizer, max_length=32):
    self.captions = captions
    self.frame_sequences = frame_sequences
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    tokenized = self.tokenizer(
        self.captions[idx],
        padding="max_length",
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
    # 去掉 tokenizer 自带的 batch 维 (1, L) -> (L,)
    item = {k: v.squeeze(0) for k, v in tokenized.items()}
    # 转为 (C, T, H, W) 方便 3D 卷积与上采样
    frames = torch.tensor(self.frame_sequences[idx], dtype=torch.float32).permute(
        3, 0, 1, 2
    )
    return item, frames


# 2. 定义可跑通的最小 Text-to-Video 生成模型
# 思路：文本特征 + 随机潜噪声 -> 投影为低分辨率时空特征图 (B, C, T, H/4, W/4)
# -> 时空 Transformer 建模帧间关系 -> 3D 上采样解码到 (B, 3, T, H, W)
class MinimalTextToVideoGenerator(nn.Module):

  def __init__(
      self,
      clip_name="openai/clip-vit-base-patch32",
      num_frames=10,
      height=64,
      width=64,
      latent_dim=128,
      freeze_text=True,
  ):
    super().__init__()
    self.num_frames = num_frames
    self.h0 = height // 4
    self.w0 = width // 4
    self.base_channels = 64

    self.text_encoder = CLIPTextModel.from_pretrained(clip_name)
    if freeze_text:
      for p in self.text_encoder.parameters():
        p.requires_grad = False

    text_hidden = self.text_encoder.config.hidden_size
    self.text_proj = nn.Sequential(
        nn.Linear(text_hidden, latent_dim),
        nn.SiLU(),
        nn.Linear(latent_dim, latent_dim),
    )

    # 将 (文本条件 + 随机噪声) 映射为初始低分辨率时空网格
    feat_dim = self.base_channels * self.h0 * self.w0
    self.noise_and_text_to_grid = nn.Sequential(
        nn.Linear(latent_dim * 2, feat_dim),
        nn.SiLU(),
    )

    # 可学习时间位置编码，建模 T 帧序列
    self.time_pos_emb = nn.Parameter(
        torch.randn(1, num_frames, feat_dim) * 0.02
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=feat_dim, nhead=4, dim_feedforward=feat_dim * 2, batch_first=True
    )
    self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # 3D 解码器：从 (B, C, T, H/4, W/4) 逐步上采样回 (B, 3, T, H, W)
    self.decoder = nn.Sequential(
        nn.Conv3d(
            self.base_channels,
            self.base_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.BatchNorm3d(self.base_channels),
        nn.SiLU(),
        nn.Upsample(
            size=(num_frames, height // 2, width // 2),
            mode="trilinear",
            align_corners=False,
        ),
        nn.Conv3d(self.base_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm3d(32),
        nn.SiLU(),
        nn.Upsample(
            size=(num_frames, height, width), mode="trilinear", align_corners=False
        ),
        nn.Conv3d(32, 3, kernel_size=3, padding=1),
        nn.Sigmoid(),  # 输出范围映射到 [0, 1]
    )

  def forward(self, text_inputs, z=None):
    batch_size = text_inputs["input_ids"].shape[0]
    device = text_inputs["input_ids"].device

    if z is None:
      z = torch.randn(batch_size, 128, device=device)

    text_out = self.text_encoder(**text_inputs).last_hidden_state.mean(dim=1)
    text_emb = self.text_proj(text_out)  # (B, 128)

    cond = torch.cat([text_emb, z], dim=-1)  # (B, 256)
    base_token = self.noise_and_text_to_grid(cond).unsqueeze(
        1
    )  # (B, 1, feat_dim)
    seq = (
        base_token.repeat(1, self.num_frames, 1) + self.time_pos_emb
    )  # (B, T, feat_dim)

    temporal_out = self.temporal_encoder(seq)  # (B, T, feat_dim)
    grid = temporal_out.view(
        batch_size, self.num_frames, self.base_channels, self.h0, self.w0
    ).permute(
        0, 2, 1, 3, 4
    )  # (B, C, T, H/4, W/4)

    video = self.decoder(grid)  # (B, 3, T, H, W)
    return video


def train_model(model, dataloader, optimizer, epochs=20, device="cpu"):
  criterion = nn.MSELoss()
  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    for tokenized, target_frames in dataloader:
      tokenized = {k: v.to(device) for k, v in tokenized.items()}
      target_frames = target_frames.to(device)  # (B, 3, T, H, W)
      z = torch.randn(target_frames.size(0), 128, device=device)

      optimizer.zero_grad()
      pred_frames = model(tokenized, z=z)
      loss = criterion(pred_frames, target_frames)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    print(
        f"Epoch {epoch + 1:02d}/{epochs:02d} | Loss:"
        f" {total_loss / len(dataloader):.6f}"
    )


@torch.no_grad()
def generate_video(model, caption, tokenizer, device="cpu", seed=42):
  model.eval()
  torch.manual_seed(seed)
  tokenized = tokenizer(
      caption,
      padding="max_length",
      truncation=True,
      max_length=32,
      return_tensors="pt",
  )
  tokenized = {k: v.to(device) for k, v in tokenized.items()}
  z = torch.randn(1, 128, device=device)
  video = model(tokenized, z=z)  # (1, 3, T, H, W)
  # 转回常见视频格式 (T, H, W, C)
  return video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()


if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_name = "openai/clip-vit-base-patch32"

  tokenizer = CLIPTokenizer.from_pretrained(model_name)
  captions, sequences = make_synthetic_videos(
      num_per_class=8, num_frames=10, size=64
  )
  dataset = VideoDataset(captions, sequences, tokenizer)
  dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

  model = MinimalTextToVideoGenerator(
      clip_name=model_name, num_frames=10, height=64, width=64
  ).to(device)
  optimizer = optim.AdamW(
      filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
  )

  train_model(model, dataloader, optimizer, epochs=20, device=device)
  generated = generate_video(
      model, "一个红色方块从左向右移动。", tokenizer, device=device
  )
  print("Generated Video Shape (T, H, W, C):", generated.shape)