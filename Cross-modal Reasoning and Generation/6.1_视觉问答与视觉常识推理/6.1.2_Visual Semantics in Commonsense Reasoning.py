import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VisualCommonsenseDataset(Dataset):
    """图文 + 问题数据集，统一 padding 以支持任意 batch_size"""

    def __init__(self, image_paths, questions, answers, processor, max_length=77):
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")

        question = self.questions[idx]
        answer = int(self.answers[idx])

        # 统一 padding 到 max_length，避免变长 batch 拼接报错
        encoded = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # 去掉单样本 batch 维度: (1, ...) -> (...)
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        return item, torch.tensor(answer, dtype=torch.long)


class VisualCommonsenseModel(nn.Module):
    """
    结合视觉特征 (image_embeds) 与问题语义特征 (text_embeds) 的常识推理分类模型。
    相比仅用 vision pooler_output，这里显式融合了“问什么”与“图里有什么”。
    """

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        num_classes=10,
        freeze_clip=False,
        dropout=0.2,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # CLIP 投影后的图像与文本向量维度一致（vit-base-patch32 下通常为 512）
        embed_dim = self.clip_model.config.projection_dim

        # 拼接 [img_feat, txt_feat, img_feat * txt_feat] -> 3 * embed_dim
        # 乘积项有助于捕捉“视觉线索与问题语义”的逐维交互
        fusion_dim = embed_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, inputs):
        outputs = self.clip_model(**inputs)

        # 使用 CLIP 对比学习对齐后的特征向量，并进行 L2 归一化
        image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

        interaction_embeds = image_embeds * text_embeds
        fused_features = torch.cat(
            [image_embeds, text_embeds, interaction_embeds], dim=-1
        )
        logits = self.classifier(fused_features)
        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, answers in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        answers = answers.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, answers)
        loss.backward()
        optimizer.step()

        batch_size = answers.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict(model, processor, image_path, question, device, max_length=77):
    model.eval()
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    encoded = processor(
        text=question,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in encoded.items()}

    logits = model(inputs)
    probs = F.softmax(logits, dim=-1)
    predicted_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_idx].item()
    return predicted_idx, confidence


def main():
    model_name = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 示例数据（实际训练请替换为真实常识推理数据集与足够样本）
    image_paths = ["image1.jpg", "image2.jpg"]
    questions = [
        "What is this object typically used for in daily life?",
        "What is the most likely consequence of this situation?",
    ]
    answers = [0, 1]

    # 若本地路径存在图片则运行演示；否则仅完成模型结构与前向自检
    if all(os.path.exists(p) for p in image_paths):
        processor = CLIPProcessor.from_pretrained(model_name)
        dataset = VisualCommonsenseDataset(image_paths, questions, answers, processor)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 建议先冻结 CLIP 或对 CLIP 与分类头使用分层学习率，避免小数据集破坏预训练对齐
        model = VisualCommonsenseModel(
            clip_model_name=model_name, num_classes=10, freeze_clip=False
        ).to(device)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.clip_model.parameters(), "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 3e-4},
            ],
            weight_decay=1e-2,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}/5 | Loss: {avg_loss:.4f}")

        test_image = "test_image.jpg"
        test_question = "What could happen next in this scene?"
        if os.path.exists(test_image):
            pred_label, conf = predict(
                model, processor, test_image, test_question, device
            )
            print(f"Predicted Answer: {pred_label} (confidence: {conf:.4f})")
        else:
            print(f"测试图片不存在: {test_image}")
    else:
        print("未检测到示例图片，跳过训练循环；可准备 image1.jpg 等文件后运行。")


if __name__ == "__main__":
    main()