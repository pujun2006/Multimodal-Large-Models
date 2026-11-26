import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.models as models
import torchvision.transforms as transforms


class GradCAM:
    """针对卷积层特征图的 Grad-CAM 解释器"""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册前向与反向钩子，捕获目标卷积层的特征图与梯度
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        probs = F.softmax(logits, dim=1)
        confidence = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward()

        # 对空间维度 (H, W) 求梯度均值，得到每个通道的权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 加权求和并经过 ReLU，只保留对目标类有正向贡献的区域
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 上采样到输入图像尺寸并归一化到 [0, 1]
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def load_or_create_sample_image(image_path: str = "sample.jpg") -> Image.Image:
    """优先读取本地真实图片；若不存在则生成含明显结构的测试图"""
    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        img = Image.new("RGB", (224, 224), color=(235, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.ellipse([55, 55, 169, 169], fill=(210, 75, 65), outline=(120, 30, 25), width=4)
        draw.rectangle([30, 175, 194, 200], fill=(60, 120, 190))
        return img


def save_overlay(rgb_img: Image.Image, cam: np.ndarray, out_path: str = "gradcam_overlay.png"):
    """将归一化热力图以红-黄-蓝风格叠加到原图上"""
    base = rgb_img.resize((cam.shape[1], cam.shape[0])).convert("RGBA")
    arr = np.asarray(base, dtype=np.float32)

    # 简单热力映射：高激活偏红黄，低激活偏冷色
    heat = np.zeros_like(arr[..., :3])
    heat[..., 0] = np.clip(cam * 255.0 * 1.2, 0, 255)
    heat[..., 1] = np.clip((cam - 0.35) * 255.0 * 1.5, 0, 255)
    heat[..., 2] = np.clip((1.0 - cam) * 180.0, 0, 255)

    alpha = (cam[..., None] * 0.55)
    blended = arr[..., :3] * (1.0 - alpha) + heat * alpha
    Image.fromarray(blended.astype(np.uint8)).save(out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    categories = weights.meta["categories"]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    pil_img = load_or_create_sample_image("cat.jpg")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # ResNet50 最后一个 Bottleneck 块的 conv3 是常用的高语义空间层
    target_layer = model.layer4[-1].conv3
    explainer = GradCAM(model, target_layer)

    try:
        # target_class=None 表示自动解释模型当前最倾向预测的类别
        cam, pred_idx, conf = explainer.generate(input_tensor, target_class=None)
    finally:
        explainer.remove_hooks()

    pred_label = categories[pred_idx]
    save_overlay(pil_img, cam, "gradcam_overlay.png")

    print(f"预测类别: {pred_label} (索引: {pred_idx}, 置信度: {conf:.4f})")
    print(f"热力图最大值位置 (y, x): {np.unravel_index(np.argmax(cam), cam.shape)}")
    print("已保存叠加结果至: gradcam_overlay.png")


if __name__ == "__main__":
    main()