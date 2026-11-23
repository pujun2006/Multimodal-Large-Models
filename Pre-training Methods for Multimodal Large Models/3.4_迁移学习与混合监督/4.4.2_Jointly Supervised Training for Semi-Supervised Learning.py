import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 1. 视角一：BERT 文本数据集
class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# 2. 视角二：TF-IDF + 线性/MLP 分类器
class TfidfClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_bert_model(model, texts, labels, tokenizer, device, epochs=3, batch_size=8, lr=2e-5):
    dataset = BertTextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_bert_probs(model, texts, tokenizer, device, batch_size=16, max_length=128):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs) if all_probs else np.empty((0, 2))


def train_tfidf_model(model, X_dense, labels, device, epochs=15, batch_size=8, lr=1e-3):
    x_tensor = torch.tensor(X_dense, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_tfidf_probs(model, X_dense, device):
    model.eval()
    if len(X_dense) == 0:
        return np.empty((0, 2))
    x_tensor = torch.tensor(X_dense, dtype=torch.float32).to(device)
    probs = torch.softmax(model(x_tensor), dim=-1)
    return probs.cpu().numpy()


def select_high_confidence_samples(probs, candidate_indices, top_k_per_class=1, min_conf=0.75):
    """
    从候选未标注样本中，按类别挑选置信度最高且高于阈值的样本。
    返回: (selected_pool_indices, pseudo_labels)
    """
    selected_pool_idx = []
    pseudo_labels = []
    num_classes = probs.shape[1]

    for c in range(num_classes):
        class_candidates = []
        for pos, pool_idx in enumerate(candidate_indices):
            conf = probs[pos, c]
            pred_label = int(np.argmax(probs[pos]))
            if pred_label == c and conf >= min_conf:
                class_candidates.append((conf, pool_idx))

        # 按置信度降序排列，取前 top_k_per_class 个
        class_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, pool_idx in class_candidates[:top_k_per_class]:
            selected_pool_idx.append(pool_idx)
            pseudo_labels.append(c)

    return selected_pool_idx, pseudo_labels


def co_train(
    labeled_texts,
    labeled_labels,
    unlabeled_texts,
    model_name="bert-base-uncased",
    rounds=4,
    pool_size=6,
    top_k_per_class=1,
    min_conf=0.60,
    seed=42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化视角 1：BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # 拟合视角 2 的 TF-IDF 特征空间（在全部可用文本上拟合词表，仅用标签样本监督训练）
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    vectorizer.fit(list(labeled_texts) + list(unlabeled_texts))
    tfidf_dim = len(vectorizer.get_feature_names_out())
    tfidf_model = TfidfClassifier(input_dim=tfidf_dim, num_classes=2).to(device)

    # 两个模型各自维护当前训练集（初始均使用标注集 L）
    texts_m1 = list(labeled_texts)
    labels_m1 = list(labeled_labels)

    texts_m2 = list(labeled_texts)
    labels_m2 = list(labeled_labels)

    unlabeled_pool = list(unlabeled_texts)

    for r in range(1, rounds + 1):
        if len(unlabeled_pool) == 0:
            break

        # 1. 分别训练两个视角模型
        train_bert_model(bert_model, texts_m1, labels_m1, tokenizer, device, epochs=3, batch_size=4)
        X_m2_train = vectorizer.transform(texts_m2).toarray()
        train_tfidf_model(tfidf_model, X_m2_train, labels_m2, device, epochs=20, batch_size=4)

        # 2. 抽取候选池 U'
        current_pool_size = min(pool_size, len(unlabeled_pool))
        candidate_indices = random.sample(range(len(unlabeled_pool)), current_pool_size)
        candidate_texts = [unlabeled_pool[i] for i in candidate_indices]

        # 3. 两个模型分别预测候选池
        probs_m1 = predict_bert_probs(bert_model, candidate_texts, tokenizer, device)
        X_cand = vectorizer.transform(candidate_texts).toarray()
        probs_m2 = predict_tfidf_probs(tfidf_model, X_cand, device)

        # 4. 各自筛选高置信度伪标签样本
        sel_idx_1, pseudo_y_1 = select_high_confidence_samples(
            probs_m1, candidate_indices, top_k_per_class=top_k_per_class, min_conf=min_conf
        )
        sel_idx_2, pseudo_y_2 = select_high_confidence_samples(
            probs_m2, candidate_indices, top_k_per_class=top_k_per_class, min_conf=min_conf
        )

        # 5. 交叉扩充训练集（互教：M1 教 M2，M2 教 M1）
        added_to_m2 = 0
        for pool_idx, y_hat in zip(sel_idx_1, pseudo_y_1):
            texts_m2.append(unlabeled_pool[pool_idx])
            labels_m2.append(y_hat)
            added_to_m2 += 1

        added_to_m1 = 0
        for pool_idx, y_hat in zip(sel_idx_2, pseudo_y_2):
            texts_m1.append(unlabeled_pool[pool_idx])
            labels_m1.append(y_hat)
            added_to_m1 += 1

        # 6. 从未标注池中移除已被打伪标签的样本
        used_indices = sorted(set(sel_idx_1) | set(sel_idx_2), reverse=True)
        for idx in used_indices:
            unlabeled_pool.pop(idx)

        print(
            f"[Round {r}] M1->M2 新增伪标签: {added_to_m2}, "
            f"M2->M1 新增伪标签: {added_to_m1}, 剩余未标注样本: {len(unlabeled_pool)}"
        )

    # 最终再用各自扩充后的训练集微调一轮
    train_bert_model(bert_model, texts_m1, labels_m1, tokenizer, device, epochs=3, batch_size=4)
    X_m2_final = vectorizer.transform(texts_m2).toarray()
    train_tfidf_model(tfidf_model, X_m2_final, labels_m2, device, epochs=20, batch_size=4)

    return bert_model, tfidf_model, tokenizer, vectorizer, device


@torch.no_grad()
def predict_ensemble(bert_model, tfidf_model, tokenizer, vectorizer, test_texts, device):
    probs_1 = predict_bert_probs(bert_model, test_texts, tokenizer, device)
    X_test = vectorizer.transform(test_texts).toarray()
    probs_2 = predict_tfidf_probs(tfidf_model, X_test, device)

    # 融合双视角预测概率（几何平均 / 乘积归一化）
    joint_probs = probs_1 * probs_2
    row_sums = joint_probs.sum(axis=1, keepdims=True)
    joint_probs = np.where(row_sums > 0, joint_probs / row_sums, 0.5 * (probs_1 + probs_2))
    preds = np.argmax(joint_probs, axis=-1)
    return preds, joint_probs


if __name__ == "__main__":
    labeled_texts = [
        "This movie is wonderful and heartwarming.",
        "I hate this film, it is a complete waste of time.",
        "Great acting and amazing story.",
        "Terrible plot and boring characters.",
    ]
    labeled_labels = [1, 0, 1, 0]

    unlabeled_texts = [
        "I really enjoyed the cinematography.",
        "Awful pacing and dull dialogue.",
        "A brilliant piece of cinema.",
        "Not worth watching at all.",
        "The soundtrack was fantastic.",
        "Completely disappointing ending.",
    ]

    bert_clf, tfidf_clf, tok, vec, dev = co_train(
        labeled_texts=labeled_texts,
        labeled_labels=labeled_labels,
        unlabeled_texts=unlabeled_texts,
        rounds=3,
        pool_size=6,
        top_k_per_class=1,
        min_conf=0.55,
    )

    test_texts = ["I love this!", "This is terrible."]
    preds, probs = predict_ensemble(bert_clf, tfidf_clf, tok, vec, test_texts, dev)
    for text, pred, prob in zip(test_texts, preds, probs):
        print(f"文本: {text!r} -> 预测类别: {pred}, 融合概率: {np.round(prob, 4)}")