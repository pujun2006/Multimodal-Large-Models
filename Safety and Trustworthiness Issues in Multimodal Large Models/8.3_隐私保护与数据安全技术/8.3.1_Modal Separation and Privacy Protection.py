import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import transforms
from cryptography.fernet import Fernet


# 1. 静态数据加密模块（用于落盘存储或跨节点传输敏感原始数据/密钥）
class DataEncryptor:
    def __init__(self, key: bytes | None = None):
        # 生产环境应从环境变量、KMS 或安全密钥库读取，切勿硬编码或随意丢弃
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_text(self, text: str) -> bytes:
        return self.cipher.encrypt(text.encode("utf-8"))

    def decrypt_text(self, token: bytes) -> str:
        return self.cipher.decrypt(token).decode("utf-8")


# 2. 模态分离编码器（模拟本地/端侧各自提取模态特征）
class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 掩码均值池化，避免 [PAD] 影响表征
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.resnet(images)


# 3. 特征级隐私扰动模块（模拟跨边界传输前的中间特征噪声扰动）
class FeaturePerturber(nn.Module):
    def __init__(self, noise_std: float = 0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            return features + torch.randn_like(features) * self.noise_std
        return features


# 4. 多模态融合与分类头（模拟下游融合/任务层）
class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim: int = 768, image_dim: int = 2048, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(self.fusion(combined))


# 5. 自定义数据集（兼容 Tensor / PIL Image / ndarray 图像输入）
class MultimodalPrivacyDataset(Dataset):
    def __init__(
        self,
        encrypted_texts: list[bytes],
        encryptor: DataEncryptor,
        images: list,
        labels: list[int],
        tokenizer: BertTokenizer,
        img_transform: transforms.Compose,
        max_length: int = 128,
    ):
        self.encrypted_texts = encrypted_texts
        self.encryptor = encryptor
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.encrypted_texts)

    def __getitem__(self, idx: int) -> dict:
        # 仅在可信本地内存域内解密当前样本用于编码
        text = self.encryptor.decrypt_text(self.encrypted_texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        img = self.images[idx]
        if isinstance(img, torch.Tensor):
            # 若已是 (C, H, W) 张量，先转为 PIL 再走标准预处理，避免 ToTensor 报错
            img_pil = transforms.ToPILImage()(img.detach().cpu().clamp(0, 1))
        else:
            img_pil = img

        image_tensor = self.img_transform(img_pil)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image_tensor,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# 6. 简化版梯度裁剪 + 噪声注入辅助函数（体现训练侧隐私保护思路；生产级 DP 建议使用 Opacus）
def apply_gradient_clip_and_noise(params: list[nn.Parameter], max_norm: float = 1.0, noise_multiplier: float = 0.01):
    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    for p in params:
        if p.grad is not None and noise_multiplier > 0:
            p.grad += torch.randn_like(p.grad) * (noise_multiplier * max_norm)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模拟敏感原始文本（先加密存储，避免明文长期驻留）
    raw_texts = [
        "Patient ID 1024: confidential diagnostic note.",
        "Internal financial record: handle with strict care.",
    ]
    # 建议通过环境变量传入固定密钥，例如：export FERNET_KEY="..."
    env_key = os.environ.get("FERNET_KEY")
    encryptor = DataEncryptor(key=env_key.encode("utf-8") if env_key else None)
    encrypted_texts = [encryptor.encrypt_text(t) for t in raw_texts]

    # 模拟图像输入（也可替换为 PIL.Image.open(...) 路径读取）
    raw_images = [torch.rand(3, 224, 224) for _ in range(len(raw_texts))]
    labels = [0, 1]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MultimodalPrivacyDataset(
        encrypted_texts=encrypted_texts,
        encryptor=encryptor,
        images=raw_images,
        labels=labels,
        tokenizer=tokenizer,
        img_transform=img_transform,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    text_encoder = TextEncoder().to(device)
    image_encoder = ImageEncoder().to(device)
    text_perturber = FeaturePerturber(noise_std=0.02).to(device)
    image_perturber = FeaturePerturber(noise_std=0.02).to(device)
    classifier = MultimodalClassifier().to(device)

    trainable_params = (
        list(text_encoder.parameters())
        + list(image_encoder.parameters())
        + list(classifier.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        text_encoder.train()
        image_encoder.train()
        classifier.train()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            # 模态分离编码 + 中间特征扰动
            text_feat = text_perturber(text_encoder(input_ids, attention_mask))
            image_feat = image_perturber(image_encoder(images))

            logits = classifier(text_feat, image_feat)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            # 训练梯度裁剪与加噪（教学/原型级演示）
            apply_gradient_clip_and_noise(trainable_params, max_norm=1.0, noise_multiplier=0.01)
            optimizer.step()

            print(f"Epoch [{epoch + 1}/2] Step [{step + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    # 验证解密闭环
    restored = [encryptor.decrypt_text(enc) for enc in encrypted_texts]
    print("Decrypted check:", restored)


if __name__ == "__main__":
    main()