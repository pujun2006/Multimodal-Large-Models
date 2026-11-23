import io
import requests
from io import BytesIO
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载权重与模型（推荐显式指定 weights）
resnet_weights = ResNet50_Weights.DEFAULT
resnet_backbone = models.resnet50(weights=resnet_weights).to(device)
resnet_backbone.fc = nn.Identity()  # 移除分类头，输出 2048 维池化特征
resnet_backbone.eval()

rcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_model = fasterrcnn_resnet50_fpn(weights=rcnn_weights).to(device)
rcnn_model.eval()
categories = rcnn_weights.meta["categories"]

# 2. 读取图片（请替换为可访问的真实图片 URL）
source = "./cat.jpg" # 也可以是 "https://example.com/cat.jpg"
if source.startswith(("http://", "https://")):
    response = requests.get(source, stream=True, timeout=10)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
else:
    image = Image.open(source).convert("RGB")




# 3. 分别准备特征提取输入与目标检测输入
cls_transform = resnet_weights.transforms()
cls_input = cls_transform(image).unsqueeze(0).to(device)

det_transform = transforms.ToTensor()  # 转为 [0.0, 1.0] 的 CHW 张量
det_input = [det_transform(image).to(device)]

with torch.no_grad():
    # 提取 2048 维视觉表征 (1, 2048)
    visual_feat = resnet_backbone(cls_input)
    # 目标检测输出为长度等于 batch 的字典列表
    detections = rcnn_model(det_input)[0]

print(f"ResNet 池化特征维度: {visual_feat.shape}")

for box, label_id, score in zip(
    detections["boxes"], detections["labels"], detections["scores"]
):
    if score.item() > 0.5:
        label_name = categories[label_id.item()]
        coords = [round(x, 1) for x in box.tolist()]
        print(
            f"类别: {label_name} (id={label_id.item()}), "
            f"置信度: {score.item():.2f}, 边界框: {coords}"
        )

# 4. 简单特征拼接示例（假设已有对齐维度的文本特征 (1, 2048)）
text_feat = torch.randn(1, visual_feat.size(1), device=device)
fused_feat = torch.cat([visual_feat, text_feat], dim=1)
print(f"拼接后特征维度: {fused_feat.shape}")
