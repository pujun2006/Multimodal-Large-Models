import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


def load_or_create_demo_image(path: str, color: Tuple[int, int, int], label: str) -> Image.Image:
  """若本地存在图片则读取并转为 RGB；若不存在则生成一张带颜色与文字的示例图，方便直接跑通"""
  if os.path.exists(path):
    return Image.open(path).convert("RGB")
  img = Image.new("RGB", (224, 224), color=color)
  draw = ImageDraw.Draw(img)
  draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=3)
  draw.text((35, 100), label[:24], fill=(255, 255, 255))
  return img


class ImageTextPairDataset(Dataset):
  """图文对数据集：返回 (PIL.Image, text)"""

  def __init__(self, pairs: List[Tuple[str, str]], fallback_colors: Optional[Dict[str, Tuple[int, int, int]]] = None):
    self.pairs = pairs
    self.fallback_colors = fallback_colors or {}

  def __len__(self) -> int:
    return len(self.pairs)

  def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
    img_path, text = self.pairs[idx]
    color = self.fallback_colors.get(img_path, (90, 130, 180))
    image = load_or_create_demo_image(img_path, color=color, label=text)
    return image, text


def make_collate_fn(processor: CLIPProcessor):
  """动态批处理：对齐并截断（CLIP 最大上下文长度为 77 tokens）"""

  def collate_fn(batch: List[Tuple[Image.Image, str]]) -> Dict[str, torch.Tensor]:
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
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


def _extract_tensor(outputs, preferred_attrs: Tuple[str, ...]) -> torch.Tensor:
  """兼容 transformers 不同版本中 get_*_features 的返回结构"""
  if isinstance(outputs, torch.Tensor):
    return outputs
  for attr in preferred_attrs:
    val = getattr(outputs, attr, None)
    if isinstance(val, torch.Tensor):
      return val
  raise TypeError(f"无法从返回类型 {type(outputs)} 中提取张量，可用属性: {dir(outputs)}")


def encode_joint_features(
    model: CLIPModel,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """提取图像与文本在共享嵌入空间中的特征"""
  kwargs_img = {"pixel_values": inputs["pixel_values"].to(device)}
  raw_image = model.get_image_features(**kwargs_img)
  image_features = _extract_tensor(raw_image, ("image_embeds", "pooler_output", "last_hidden_state"))
  if image_features.ndim == 3:
    image_features = image_features[:, 0, :]

  if "input_ids" in inputs:
    kwargs_txt = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }
    raw_text = model.get_text_features(**kwargs_txt)
    text_features = _extract_tensor(raw_text, ("text_embeds", "pooler_output", "last_hidden_state"))
    if text_features.ndim == 3:
      text_features = text_features[:, -1, :]
  else:
    text_features = torch.empty(0, image_features.size(-1), device=device)

  if normalize:
    image_features = F.normalize(image_features, p=2, dim=-1)
    if text_features.numel() > 0:
      text_features = F.normalize(text_features, p=2, dim=-1)

  return image_features, text_features


class SymmetricCLIPLoss(nn.Module):
  """CLIP 双向对称对比损失，复用模型可学习 logit_scale 并做安全裁剪"""

  def __init__(self, max_logit_scale: float = 100.0):
    super().__init__()
    self.max_log_scale = float(np.log(max_logit_scale))

  def forward(
      self,
      model: CLIPModel,
      image_features: torch.Tensor,
      text_features: torch.Tensor,
  ) -> torch.Tensor:
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    with torch.no_grad():
      model.logit_scale.clamp_(max=self.max_log_scale)
    logit_scale = model.logit_scale.exp()

    # [B, B] 缩放余弦相似度矩阵
    logits_per_image = logit_scale * (image_features @ text_features.T)
    logits_per_text = logits_per_image.T

    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def fine_tune_clip_contrastive(
    model: CLIPModel,
    processor: CLIPProcessor,
    train_pairs: List[Tuple[str, str]],
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    fallback_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> None:
  """双向对称对比微调（建议在真实任务中使用较大 batch_size 与充足负样本）"""
  dataset = ImageTextPairDataset(train_pairs, fallback_colors=fallback_colors)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=make_collate_fn(processor),
  )

  criterion = SymmetricCLIPLoss(max_logit_scale=100.0)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  model.to(device)

  for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch_inputs in dataloader:
      optimizer.zero_grad()
      image_features, text_features = encode_joint_features(model, batch_inputs, device, normalize=True)
      loss = criterion(model, image_features, text_features)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      steps += 1

    print(f"[对比微调] Epoch {epoch}/{epochs} - 平均对称损失: {total_loss / max(steps, 1):.4f}")


@torch.no_grad()
def build_zero_shot_classifier(
    model: CLIPModel,
    processor: CLIPProcessor,
    class_names: List[str],
    templates: List[str],
    device: torch.device,
) -> torch.Tensor:
  """
  用多条自然语言模板为每个类别生成归一化权重向量。
  返回形状: [num_classes, embed_dim]
  """
  model.eval()
  class_weights = []

  for name in class_names:
    prompts = [tpl.format(name) for tpl in templates]
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    text_outputs = model.get_text_features(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device),
    )
    text_embeds = _extract_tensor(text_outputs, ("text_embeds", "pooler_output", "last_hidden_state"))
    if text_embeds.ndim == 3:
      text_embeds = text_embeds[:, -1, :]

    # 多模板平均后再次做 L2 归一化
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    mean_embed = F.normalize(text_embeds.mean(dim=0, keepdim=True), p=2, dim=-1)
    class_weights.append(mean_embed.squeeze(0))

  return torch.stack(class_weights, dim=0)


@torch.no_grad()
def zero_shot_predict(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: List[Image.Image],
    class_names: List[str],
    class_weights: torch.Tensor,
    device: torch.device,
) -> List[Tuple[str, float, np.ndarray]]:
  """零样本推理：计算图像特征与文本类别向量的余弦相似度及概率分布"""
  model.eval()
  img_inputs = processor(images=images, return_tensors="pt", padding=True)
  image_features, _ = encode_joint_features(model, img_inputs, device, normalize=True)

  logit_scale = model.logit_scale.exp()
  logits = logit_scale * (image_features @ class_weights.T)  # [B, num_classes]
  probs = F.softmax(logits, dim=-1).cpu().numpy()

  results = []
  for i in range(len(images)):
    pred_idx = int(np.argmax(probs[i]))
    results.append((class_names[pred_idx], float(probs[i, pred_idx]), probs[i]))
  return results


class CLIPLinearProbe(nn.Module):
  """线性探测：冻结 CLIP 主干，仅训练图像特征上的线性分类层"""

  def __init__(self, clip_model: CLIPModel, num_classes: int, freeze_backbone: bool = True):
    super().__init__()
    self.clip_model = clip_model
    if freeze_backbone:
      for param in self.clip_model.parameters():
        param.requires_grad = False

    embed_dim = clip_model.config.projection_dim
    self.head = nn.Linear(embed_dim, num_classes)

  def forward(self, pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = {"pixel_values": pixel_values}
    image_features, _ = encode_joint_features(self.clip_model, inputs, device, normalize=True)
    return self.head(image_features)


def train_linear_probe(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    train_samples: List[Tuple[Image.Image, int]],
    num_classes: int,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> CLIPLinearProbe:
  """训练线性分类头（仅更新 head 参数）"""
  probe = CLIPLinearProbe(clip_model, num_classes=num_classes, freeze_backbone=True).to(device)
  optimizer = torch.optim.AdamW(probe.head.parameters(), lr=lr, weight_decay=1e-4)

  images = [item[0] for item in train_samples]
  labels = torch.tensor([item[1] for item in train_samples], dtype=torch.long, device=device)
  pixel_values = processor(images=images, return_tensors="pt", padding=True)["pixel_values"]

  for epoch in range(1, epochs + 1):
    probe.train()
    optimizer.zero_grad()
    logits = probe(pixel_values, device)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"[线性探测] Epoch {epoch}/{epochs} - 分类交叉熵损失: {loss.item():.4f}")

  return probe


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"运行设备: {device}")

  model_id = "openai/clip-vit-base-patch32"
  model = CLIPModel.from_pretrained(model_id)
  processor = CLIPProcessor.from_pretrained(model_id)

  fallback_colors = {
      "cat.jpg": (220, 120, 60),
      "dog.jpg": (70, 130, 180),
      "bird.jpg": (210, 180, 40),
  }

  # 1. 双向对称对比微调示例（建议与 clip-vit-base-patch32 搭配英文描述效果更佳）
  train_pairs = [
      ("cat.jpg", "a tabby cat with green eyes sits on a light-colored ledge"),
      ("dog.jpg", "a yellow dog stands on a sandy beach"),
      ("bird.jpg", "a colorful bird perched on a branch"),
      ("cat.jpg", "a close-up photo of a cat"),
  ]
  fine_tune_clip_contrastive(
      model=model,
      processor=processor,
      train_pairs=train_pairs,
      device=device,
      epochs=2,
      batch_size=4,
      lr=5e-6,
      fallback_colors=fallback_colors,
  )

  # 2. 零样本分类迁移（无需分类训练头，使用多模板文本特征作为分类权重）
  class_names = ["cat", "dog", "bird"]
  templates = [
      "a photo of a {}",
      "a picture of a {}",
      "a close-up photo of a {}",
  ]
  class_weights = build_zero_shot_classifier(model, processor, class_names, templates, device)

  test_images = [
      load_or_create_demo_image("cat.jpg", (220, 120, 60), "test cat"),
      load_or_create_demo_image("dog.jpg", (70, 130, 180), "test dog"),
  ]
  zs_results = zero_shot_predict(model, processor, test_images, class_names, class_weights, device)
  for idx, (pred_label, conf, probs) in enumerate(zs_results):
    print(f"[零样本推理] 测试图 {idx + 1} -> 预测: {pred_label} (置信度: {conf:.4f}), 各类概率: {np.round(probs, 4)}")

  # 3. 线性探测迁移（冻结 CLIP 主干，仅训练线性分类头）
  probe_train_data = [
      (load_or_create_demo_image("cat.jpg", (220, 120, 60), "cat"), 0),
      (load_or_create_demo_image("dog.jpg", (70, 130, 180), "dog"), 1),
      (load_or_create_demo_image("bird.jpg", (210, 180, 40), "bird"), 2),
  ]
  trained_probe = train_linear_probe(
      clip_model=model,
      processor=processor,
      train_samples=probe_train_data,
      num_classes=len(class_names),
      device=device,
      epochs=5,
      lr=1e-3,
  )

  trained_probe.eval()
  with torch.no_grad():
    sample_pv = processor(images=[test_images[1]], return_tensors="pt", padding=True)["pixel_values"]
    probe_logits = trained_probe(sample_pv, device)
    probe_pred = int(torch.argmax(probe_logits, dim=-1).item())
    print(f"[线性探测推理] 测试图 2 -> 预测类别: {class_names[probe_pred]}")