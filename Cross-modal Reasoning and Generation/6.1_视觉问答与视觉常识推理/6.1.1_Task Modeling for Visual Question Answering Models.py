import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


class AnswerVocabulary:
    """答案词表管理器，负责文本答案与类别索引的双向映射。"""

    def __init__(self, max_size: int = 1000, unk_token: str = "<unk>"):
        self.max_size = max_size
        self.unk_token = unk_token
        self.answer_to_idx: Dict[str, int] = {unk_token: 0}
        self.idx_to_answer: List[str] = [unk_token]

    def build(self, answers: List[str]) -> None:
        freq: Dict[str, int] = {}
        for ans in answers:
            norm_ans = ans.strip().lower()
            if norm_ans and norm_ans != self.unk_token:
                freq[norm_ans] = freq.get(norm_ans, 0) + 1

        # 按出现频次降序截取词表
        sorted_answers = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for ans, _ in sorted_answers[: max(0, self.max_size - 1)]:
            if ans not in self.answer_to_idx:
                self.answer_to_idx[ans] = len(self.idx_to_answer)
                self.idx_to_answer.append(ans)

    def encode(self, answer: str) -> int:
        return self.answer_to_idx.get(answer.strip().lower(), 0)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.idx_to_answer):
            return self.idx_to_answer[idx]
        return self.unk_token

    def __len__(self) -> int:
        return len(self.idx_to_answer)


class VQADataset(Dataset):
    """视觉问答数据集：加载图像与问题，返回去掉 batch 维度的张量及答案标签。"""

    def __init__(
        self,
        image_paths: List[str],
        questions: List[str],
        answers: List[str],
        processor: CLIPProcessor,
        vocab: AnswerVocabulary,
        max_text_length: int = 77,
    ):
        if not (len(image_paths) == len(questions) == len(answers)):
            raise ValueError("image_paths、questions 与 answers 的长度必须一致。")

        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.processor = processor
        self.vocab = vocab
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")

        question = self.questions[idx]
        label_idx = self.vocab.encode(self.answers[idx])

        encoded = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


def build_collate_fn(pad_token_id: int):
    """构造动态 padding 的 collate_fn，对齐同批次内不同长度的问题序列。"""

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        labels_batch = []

        for item in batch:
            cur_len = item["input_ids"].size(0)
            pad_len = max_len - cur_len

            if pad_len > 0:
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=item["input_ids"].dtype)
                pad_mask = torch.zeros((pad_len,), dtype=item["attention_mask"].dtype)
                input_ids = torch.cat([item["input_ids"], pad_ids], dim=0)
                attention_mask = torch.cat([item["attention_mask"], pad_mask], dim=0)
            else:
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(item["pixel_values"])
            labels_batch.append(item["label"])

        model_inputs = {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "pixel_values": torch.stack(pixel_values_batch, dim=0),
        }
        labels = torch.stack(labels_batch, dim=0)
        return model_inputs, labels

    return collate_fn


class MultimodalVQAModel(nn.Module):
    """
    基于 CLIP 视觉与文本编码器的分类式 VQA 模型：
    1. 提取图像 patch/全局特征与文本 token/全局特征并映射到共享维度；
    2. 使用多头交叉注意力让问题特征查询相关视觉特征；
    3. 拼接融合特征后通过 MLP 分类头输出答案分布。
    """

    def __init__(
        self,
        num_classes: int,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip: bool = False,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        proj_dim = self.clip_model.config.projection_dim
        vision_dim = self.clip_model.config.vision_config.hidden_size
        text_dim = self.clip_model.config.text_config.hidden_size

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 将视觉 patch 隐藏状态与文本 token 隐藏状态统一投影到 proj_dim
        self.vision_token_proj = nn.Linear(vision_dim, proj_dim)
        self.text_token_proj = nn.Linear(text_dim, proj_dim)

        # 交叉注意力：query 为问题 token 表征，key/value 为图像 patch 表征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        # 融合特征维度：全局图像向量 + 全局文本向量 + 交叉注意力池化向量 = 3 * proj_dim
        fused_dim = proj_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 全局投影特征（shape: [B, proj_dim]）
        global_img_feat = self.clip_model.visual_projection(vision_outputs.pooler_output)
        global_txt_feat = self.clip_model.text_projection(text_outputs.pooler_output)

        # 序列级特征（shape: [B, N_vis, proj_dim] 与 [B, N_txt, proj_dim]）
        vis_seq_feat = self.vision_token_proj(vision_outputs.last_hidden_state)
        txt_seq_feat = self.text_token_proj(text_outputs.last_hidden_state)

        # 交叉注意力融合
        attn_out, _ = self.cross_attn(
            query=txt_seq_feat,
            key=vis_seq_feat,
            value=vis_seq_feat,
        )
        attn_out = self.attn_norm(txt_seq_feat + attn_out)

        # 使用 attention_mask 对有效文本 token 做平均池化
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_attn_feat = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        fused_feat = torch.cat([global_img_feat, global_txt_feat, pooled_attn_feat], dim=-1)
        logits = self.classifier(fused_feat)
        return logits


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, labels in dataloader:
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(**batch_inputs)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, labels in dataloader:
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = labels.to(device)

        logits = model(**batch_inputs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def predict_answer(
    model: nn.Module,
    processor: CLIPProcessor,
    vocab: AnswerVocabulary,
    image_path: str,
    question: str,
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    model.eval()
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    inputs = processor(text=question, images=image, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    k = min(top_k, probs.size(0))
    top_probs, top_indices = torch.topk(probs, k=k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append((vocab.decode(idx), float(prob)))
    return results


def _prepare_sample_data(output_dir: str = "_vqa_demo_data") -> Tuple[List[str], List[str], List[str]]:
    """生成少量本地合成图像与示例问答对，便于无外部数据集时直接跑通全流程。"""
    os.makedirs(output_dir, exist_ok=True)
    specs = [
        ("sample_red.png", (220, 40, 40), (40, 40, 40), "What color is the background?", "red"),
        ("sample_blue.png", (40, 90, 220), (250, 220, 50), "What color is the background?", "blue"),
        ("sample_green.png", (45, 175, 75), (255, 255, 255), "What color is the background?", "green"),
        ("sample_yellow.png", (240, 200, 30), (30, 30, 30), "What color is the background?", "yellow"),
    ]

    image_paths, questions, answers = [], [], []
    for filename, bg_color, fg_color, q, a in specs:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            img = Image.new("RGB", (224, 224), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.ellipse([62, 62, 162, 162], fill=fg_color)
            img.save(path)
        image_paths.append(path)
        questions.append(q)
        answers.append(a)

    return image_paths, questions, answers


def main(
    image_paths: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
) -> None:
    if image_paths is None or questions is None or answers is None:
        image_paths, questions, answers = _prepare_sample_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    vocab = AnswerVocabulary(max_size=1000)
    vocab.build(answers)

    dataset = VQADataset(
        image_paths=image_paths,
        questions=questions,
        answers=answers,
        processor=processor,
        vocab=vocab,
    )

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(pad_token_id=pad_token_id),
    )

    model = MultimodalVQAModel(
        num_classes=len(vocab),
        clip_model_name=clip_model_name,
        freeze_clip=False,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

    top_preds = predict_answer(
        model=model,
        processor=processor,
        vocab=vocab,
        image_path=image_paths[0],
        question=questions[0],
        device=device,
        top_k=3,
    )
    print(f"测试问题: {questions[0]} | Top 预测: {top_preds}")


if __name__ == "__main__":
    main()