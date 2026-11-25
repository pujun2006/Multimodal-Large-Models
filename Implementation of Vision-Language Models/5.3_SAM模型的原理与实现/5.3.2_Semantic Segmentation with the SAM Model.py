import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import SamModel, SamProcessor


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
  """计算二值掩码的 IoU 与 Dice 系数"""
  pred = pred_mask.astype(bool)
  gt = gt_mask.astype(bool)
  inter = np.logical_and(pred, gt).sum()
  union = np.logical_or(pred, gt).sum()
  gt_sum = gt.sum()
  pred_sum = pred.sum()

  iou = (
      1.0 if union == 0 and inter == 0 else (0.0 if union == 0 else inter / union)
  )
  dice = (
      1.0
      if (gt_sum + pred_sum) == 0
      else (2.0 * inter) / (gt_sum + pred_sum)
  )
  return {"iou": float(iou), "dice": float(dice)}


def build_sample_data(h: int = 512, w: int = 512):
  """生成一张含矩形目标的示例图像、边界框提示与真实掩码"""
  img = Image.new("RGB", (w, h), color=(40, 44, 52))
  gt = np.zeros((h, w), dtype=np.uint8)

  x1, y1, x2, y2 = w // 4, h // 4, (w * 3) // 4, (h * 3) // 4
  draw = ImageDraw.Draw(img)
  draw.rectangle([x1, y1, x2, y2], fill=(220, 90, 70), outline=(255, 220, 120), width=3)
  gt[y1 : y2 + 1, x1 : x2 + 1] = 1

  # 边界框提示格式: [[[x1, y1, x2, y2]]]
  input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]
  return img, input_boxes, gt


def run_sam_evaluation(
    model_id: str = "facebook/sam-vit-base",
    image_path: str | None = None,
    input_boxes: list | None = None,
    gt_mask: np.ndarray | None = None,
    runs: int = 5,
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"运行设备: {device}")

  if image_path is not None:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if input_boxes is None:
      input_boxes = [[[w * 0.1, h * 0.1, w * 0.9, h * 0.9]]]
    if gt_mask is None:
      gt_mask = np.zeros((h, w), dtype=np.uint8)
  else:
    image, input_boxes, gt_mask = build_sample_data(512, 512)

  processor = SamProcessor.from_pretrained(model_id)
  model = SamModel.from_pretrained(model_id).to(device)
  model.eval()

  inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
  inputs = {
      k: v.to(device) if isinstance(v, torch.Tensor) else v
      for k, v in inputs.items()
  }

  # 预热（避免首次 CUDA 初始化/内核编译干扰计时）
  with torch.no_grad():
    _ = model(**inputs, multimask_output=False)

  if device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

  latencies_ms = []
  last_outputs = None

  with torch.no_grad():
    for _ in range(runs):
      t0 = time.perf_counter()
      outputs = model(**inputs, multimask_output=False)
      if device.type == "cuda":
        torch.cuda.synchronize()
      latencies_ms.append((time.perf_counter() - t0) * 1000.0)
      last_outputs = outputs

  # 将预测掩码还原至原始图像分辨率
  pred_masks = processor.image_processor.post_process_masks(
      last_outputs.pred_masks.cpu(),
      inputs["original_sizes"].cpu(),
      inputs["reshaped_input_sizes"].cpu(),
  )
  # pred_masks[0] 形状为 [num_boxes, num_masks, H, W]
  best_mask = (pred_masks[0][0, 0].numpy() > 0.0).astype(np.uint8)
  predicted_iou = float(last_outputs.iou_scores[0, 0, 0].item())

  metrics = compute_metrics(best_mask, gt_mask)
  avg_ms = float(np.mean(latencies_ms))
  fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

  print(f"掩码尺寸: {best_mask.shape}")
  print(f"平均推理耗时: {avg_ms:.2f} ms (约 {fps:.2f} FPS)")
  if device.type == "cuda":
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"峰值显存占用: {peak_mb:.1f} MB")
  print(f"SAM 预测 IoU 置信度: {predicted_iou:.4f}")
  print(f"与真值对比 -> IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")

  return best_mask, metrics


if __name__ == "__main__":
  # 若本地有真实图像与标注，可传入 image_path、input_boxes 与 gt_mask
  run_sam_evaluation(model_id="facebook/sam-vit-base", runs=5)