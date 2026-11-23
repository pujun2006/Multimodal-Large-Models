import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 示例使用开源 Instruct 模型（实际使用时替换为本地或可访问的模型名）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)


def enhance_context(context: str, additional_info: str | None = None, knowledge_items: list[str] | None = None) -> str:
    """输入增强：整合背景、附加信息与外部知识片段"""
    parts = []
    if context:
        parts.append(f"[背景]\n{context.strip()}")
    if additional_info:
        parts.append(f"[补充说明]\n{additional_info.strip()}")
    if knowledge_items:
        formatted_kb = "\n".join(f"- {item.strip()}" for item in knowledge_items if item.strip())
        if formatted_kb:
            parts.append(f"[参考知识]\n{formatted_kb}")
    return "\n\n".join(parts)


def build_prompt(question: str, augmented_context: str) -> str:
    """提示模板设计：明确任务规则、上下文槽位与输出要求"""
    return (
        "你是一名技术分析助手。请结合以下参考信息回答问题，要求条理清晰、准确客观；"
        "若参考信息未涵盖的内容，请明确说明。\n\n"
        f"{augmented_context}\n\n"
        f"[问题]\n{question.strip()}\n\n"
        "[回答]\n"
    )


context = "多模态模型能够结合文本与图像信息执行复杂的任务。"
question = "多模态模型的主要优点是什么？"
additional_info = "可以通过视觉信息辅助语言理解。"
knowledge_base = ["多模态技术能够提升任务的鲁棒性与准确性。"]

augmented_ctx = enhance_context(context, additional_info, knowledge_base)
prompt_text = build_prompt(question, augmented_ctx)

# 若模型支持 chat_template，可按对话格式包裹；否则直接使用 prompt_text
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt_text}]
    formatted_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
else:
    formatted_input = prompt_text

inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
    )

# 仅解码新生成的 token
generated_ids = outputs[0][input_len:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("生成的答案：\n", answer)