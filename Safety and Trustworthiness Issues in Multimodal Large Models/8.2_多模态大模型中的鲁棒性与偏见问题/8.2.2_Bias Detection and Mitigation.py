import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=32):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    enc = self.tokenizer(
        self.texts[idx],
        max_length=self.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "label": torch.tensor(self.labels[idx], dtype=torch.long),
    }


# 示例：标签与代词解耦，避免把敏感词直接等价为标签
texts = [
    "She is a skilled doctor.",
    "He is a skilled doctor.",
    "She is a dedicated nurse.",
    "He is a dedicated nurse.",
    "She is a senior engineer.",
    "He is a senior engineer.",
    "She is a caring teacher.",
    "He is a caring teacher.",
]
labels = [1, 1, 0, 0, 1, 1, 0, 0]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


def train_with_fgm(model, data_loader, optimizer, epsilon=1.0, epochs=2):
  embedding_layer = model.get_input_embeddings()
  model.train()

  for epoch in range(epochs):
    total_loss = 0.0
    for batch in data_loader:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["label"].to(device)

      optimizer.zero_grad()

      # 1. 正常前向传播（基于词嵌入）
      inputs_embeds = embedding_layer(input_ids).detach()
      inputs_embeds.requires_grad_(True)
      outputs = model(
          inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
      )
      loss = outputs.loss
      loss.backward()

      # 2. 对 Embedding 施加 FGM 扰动
      grad = inputs_embeds.grad
      norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
      perturbation = epsilon * grad / norm
      perturbed_embeds = (inputs_embeds + perturbation).detach()

      # 3. 对抗样本前向传播
      adv_outputs = model(
          inputs_embeds=perturbed_embeds,
          attention_mask=attention_mask,
          labels=labels,
      )
      adv_loss = adv_outputs.loss
      adv_loss.backward()

      optimizer.step()
      total_loss += (loss.item() + adv_loss.item()) / 2

    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(loader):.4f}")


def evaluate_counterfactual_pairs(model, tokenizer, pairs):
  model.eval()
  with torch.no_grad():
    for female_text, male_text in pairs:
      probs = []
      for t in (female_text, male_text):
        enc = tokenizer(t, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        logits = model(**enc).logits
        p1 = torch.softmax(logits, dim=-1)[0, 1].item()
        probs.append(p1)
      diff = abs(probs[0] - probs[1])
      print(
          f"'{female_text}' (p1={probs[0]:.3f}) vs '{male_text}'"
          f" (p1={probs[1]:.3f}) | 差异={diff:.3f}"
      )


train_with_fgm(model, loader, optimizer, epsilon=0.5, epochs=2)
test_pairs = [
    ("She is a doctor.", "He is a doctor."),
    ("She is a nurse.", "He is a nurse."),
]
evaluate_counterfactual_pairs(model, tokenizer, test_pairs)