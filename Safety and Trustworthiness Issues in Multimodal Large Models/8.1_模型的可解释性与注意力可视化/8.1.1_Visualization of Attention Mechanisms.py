import math
import matplotlib
matplotlib.use("TkAgg")  # 若弹窗失败可改为 "Qt5Agg" 或仅存图用 "Agg"
import matplotlib.pyplot as plt
import torch
from transformers import BertModel, BertTokenizer


def load_bert_and_tokenize(text: str, model_name: str = "bert-base-uncased"):
    """加载模型与分词器，并在无梯度模式下提取注意力权重。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions 是长度为 num_layers 的元组
    # 每个元素形状为 (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return attentions, tokens


def slice_tokens_and_matrix(matrix: torch.Tensor, tokens: list[str], drop_special: bool = True):
    """可选剔除首尾特殊词元（[CLS] 与 [SEP]），避免注意力汇（attention sink）压制其余权重。"""
    if drop_special and len(tokens) > 2:
        tokens = tokens[1:-1]
        matrix = matrix[1:-1, 1:-1]
    return matrix.cpu().numpy(), tokens


def plot_single_head(
    attentions: tuple[torch.Tensor, ...],
    tokens: list[str],
    layer_idx: int,
    head_idx: int,
    drop_special: bool = True,
    save_path: str | None = None,
):
    """绘制指定层与指定注意力头的热力图（行：Query，列：Key）。"""
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx 必须在 [0, {num_layers - 1}] 范围内，当前为 {layer_idx}")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx 必须在 [0, {num_heads - 1}] 范围内，当前为 {head_idx}")

    # 取 batch 0 的注意力矩阵：形状 (seq_len, seq_len)，每一行和为 1（Query -> Key）
    raw_matrix = attentions[layer_idx][0, head_idx]
    matrix, display_tokens = slice_tokens_and_matrix(raw_matrix, tokens, drop_special=drop_special)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)

    ticks = range(len(display_tokens))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(display_tokens, fontsize=10)

    ax.set_xlabel("Key Tokens (Attended To)", fontsize=11)
    ax.set_ylabel("Query Tokens (Attending From)", fontsize=11)
    suffix = " (excl. [CLS]/[SEP])" if drop_special else " (incl. [CLS]/[SEP])"
    ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}{suffix}", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_layer_heads_grid(
    attentions: tuple[torch.Tensor, ...],
    tokens: list[str],
    layer_idx: int,
    drop_special: bool = True,
    cols: int = 4,
    save_path: str | None = None,
):
    """在同一张画布中对比某一层全部注意力头的分布，避免跨头平均丢失分工。"""
    num_layers = len(attentions)
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx 必须在 [0, {num_layers - 1}] 范围内，当前为 {layer_idx}")

    layer_tensor = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
    num_heads = layer_tensor.shape[0]
    rows = math.ceil(num_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.4), squeeze=False)
    sample_matrix, display_tokens = slice_tokens_and_matrix(
        layer_tensor[0], tokens, drop_special=drop_special
    )
    n_tok = len(display_tokens)

    last_im = None
    for h in range(num_heads):
        r, c = divmod(h, cols)
        ax = axes[r][c]
        mat, _ = slice_tokens_and_matrix(layer_tensor[h], tokens, drop_special=drop_special)
        last_im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"Head {h + 1}", fontsize=10)
        ax.set_xticks(range(n_tok))
        ax.set_yticks(range(n_tok))
        ax.set_xticklabels(display_tokens, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(display_tokens, fontsize=7)

    # 隐藏多余子图
    for idx in range(num_heads, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"All Heads in Layer {layer_idx + 1} (Row: Query, Col: Key)", fontsize=13, y=1.02)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label="Attention Weight")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog"
    attentions, tokens = load_bert_and_tokenize(sample_text)

    # 1. 查看第 1 层第 1 头、第 6 层第 5 头
    plot_single_head(attentions, tokens, layer_idx=0, head_idx=0, drop_special=True)
    plot_single_head(attentions, tokens, layer_idx=5, head_idx=4, drop_special=True)

    # 2. 对比第 6 层全部 12 个注意力头的分工
    plot_layer_heads_grid(attentions, tokens, layer_idx=5, drop_special=True, cols=4)