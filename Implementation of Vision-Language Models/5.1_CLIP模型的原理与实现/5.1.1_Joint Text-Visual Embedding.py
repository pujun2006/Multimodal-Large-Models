from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


class ImageTextDataset(Dataset):
  """图文配对数据集：每个样本是一张图像及其对应的文本描述"""

  def __init__(self, pairs: List[Tuple[str, str]]):
    # pairs: [(image_path, text), ...]
    self.pairs = pairs

  def __len__(self) -> int:
    return len(self.pairs)

  def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
    img_path, text = self.pairs[idx]
    image = Image.open(img_path).convert("RGB")
    return image, text


def make_collate_fn(processor: CLIPProcessor):
  """动态将一个 batch 的 PIL 图像与文本编码为张量"""

  def collate_fn(batch: List[Tuple[Image.Image, str]]):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    # truncation=True 防止超长文本超过 CLIP 最大长度（77 tokens）
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    return inputs

  return collate_fn


def _extract_feature_tensor(outputs, preferred_attrs: Tuple[str, ...]) -> torch.Tensor:
  """兼容 transformers 不同版本中 get_text_features / get_image_features 的返回类型"""
  if isinstance(outputs, torch.Tensor):
    return outputs
  for attr in preferred_attrs:
    val = getattr(outputs, attr, None)
    if isinstance(val, torch.Tensor):
      return val
  raise TypeError(f"无法从类型 {type(outputs)} 中提取特征张量，可用属性: {dir(outputs)}")


def compute_joint_embeddings(
    model: CLIPModel,
    inputs: dict,
    device: torch.device,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """将图像与文本分别映射到联合嵌入空间"""
  input_ids = inputs["input_ids"].to(device)
  attention_mask = inputs["attention_mask"].to(device)
  pixel_values = inputs["pixel_values"].to(device)

  raw_text = model.get_text_features(
      input_ids=input_ids, attention_mask=attention_mask
  )
  raw_image = model.get_image_features(pixel_values=pixel_values)

  text_features = _extract_feature_tensor(
      raw_text, ("text_embeds", "pooler_output", "last_hidden_state")
  )
  image_features = _extract_feature_tensor(
      raw_image, ("image_embeds", "pooler_output", "last_hidden_state")
  )

  # 若取到的是序列特征 [B, seq_len, D]，退化兜底取第一个/最后一个 token
  if text_features.ndim == 3:
    text_features = text_features[:, -1, :]
  if image_features.ndim == 3:
    image_features = image_features[:, 0, :]

  if normalize:
    text_features = F.normalize(text_features, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)

  return text_features, image_features


class CLIPLoss(nn.Module):
  """CLIP 双向对称对比损失，使用模型自带可学习 logit_scale"""

  def __init__(self, max_logit_scale: float = 100.0):
    super().__init__()
    self.max_log_scale = np.log(max_logit_scale)

  def forward(
      self,
      model: CLIPModel,
      text_features: torch.Tensor,
      image_features: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 确保特征已归一化，使点积严格等于余弦相似度
    text_features = F.normalize(text_features, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)

    # 裁剪对数温度缩放系数，防止训练中缩放过大导致梯度不稳定
    with torch.no_grad():
      model.logit_scale.clamp_(max=self.max_log_scale)
    logit_scale = model.logit_scale.exp()

    # logits_per_image: [B, B]，第 i 行表示第 i 张图与 batch 内 B 条文本的缩放相似度
    logits_per_image = logit_scale * (image_features @ text_features.T)
    logits_per_text = logits_per_image.T

    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)

    return loss, logits_per_image, logits_per_text


@torch.no_grad()
def evaluate_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: List[str],
    texts: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
  """推理评估：返回余弦相似度矩阵（范围 [-1, 1]）与缩放后的 logits 矩阵"""
  model.eval()
  images = [Image.open(p).convert("RGB") for p in image_paths]
  inputs = processor(
      text=texts,
      images=images,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=77,
  )

  text_features, image_features = compute_joint_embeddings(
      model, inputs, device, normalize=True
  )
  # [num_images, num_texts] 余弦相似度矩阵
  cos_sim_img_to_txt = (image_features @ text_features.T).cpu().numpy()

  logit_scale = model.logit_scale.exp().item()
  scaled_logits = cos_sim_img_to_txt * logit_scale
  return cos_sim_img_to_txt, scaled_logits


def train_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    train_pairs: List[Tuple[str, str]],
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
):
  """微调 CLIP 双编码器联合嵌入空间"""
  dataset = ImageTextDataset(train_pairs)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=make_collate_fn(processor),
  )

  criterion = CLIPLoss(max_logit_scale=100.0)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

  model.to(device)

  for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_inputs in dataloader:
      optimizer.zero_grad()

      text_features, image_features = compute_joint_embeddings(
          model, batch_inputs, device, normalize=True
      )
      loss, _, _ = criterion(model, text_features, image_features)

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch [{epoch}/{epochs}] - 平均双向对比损失: {avg_loss:.4f}")


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"使用设备: {device}")

  model_name = "openai/clip-vit-base-patch32"
  clip_model = CLIPModel.from_pretrained(model_name)
  clip_processor = CLIPProcessor.from_pretrained(model_name)

  # 1. 推理前基线测试（建议替换为你的真实图片路径与对应描述）
  sample_images = ["dog.jpg", "landscape.jpg"]
  sample_texts = ["a cute dog", "a beautiful landscape"]
  try:
    init_cos_sim, _ = evaluate_similarity(
        clip_model, clip_processor, sample_images, sample_texts, device
    )
    print("微调前 图文余弦相似度矩阵（行=图像，列=文本）:")
    print(np.round(init_cos_sim, 4))
  except FileNotFoundError as e:
    print(f"跳过示例推理（未找到图片文件）: {e}")

  # 2. 联合嵌入微调示例
  # 注意：对比学习需要足够的 batch_size 提供负样本；数据量较少时可适当调小 batch_size
  train_data = [
      ("cat.jpg", "a cat sitting on a sofa"),
      ("car.jpg", "a red car parked on the street"),
      ("dog.jpg", "a cute dog"),
      ("landscape.jpg", "a beautiful landscape"),
  ]
  try:
    train_clip(
        model=clip_model,
        processor=clip_processor,
        train_pairs=train_data,
        epochs=3,
        batch_size=4,
        lr=5e-6,
        device=device,
    )

    # 3. 微调后推理验证
    eval_images = ["building.jpg", "bird.jpg"]
    eval_texts = ["a tall modern building", "a bird flying in the sky"]
    final_cos_sim, _ = evaluate_similarity(
        clip_model, clip_processor, eval_images, eval_texts, device
    )
    print("微调后 图文余弦相似度矩阵（行=图像，列=文本）:")
    print(np.round(final_cos_sim, 4))
  except FileNotFoundError as e:
    print(f"跳过微调/评估（未找到训练或测试图片文件）: {e}")