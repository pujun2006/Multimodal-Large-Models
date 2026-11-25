import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """示例网络：包含线性层、归一化/激活与输出层，便于演示图优化与动态 batch。"""

    def __init__(self, in_features: int = 10, hidden_features: int = 20, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def export_to_onnx(
    model: nn.Module,
    onnx_path: str = "simple_model.onnx",
    in_features: int = 10,
    opset_version: int = 17,
) -> str:
    """导出 PyTorch 模型为 ONNX，并执行结构合法性检查。"""
    model.eval()
    dummy_input = torch.randn(1, in_features, dtype=torch.float32)

    # 兼容不同 PyTorch 版本的导出参数写法：优先尝试新版 dynamo 导出，失败时回退传统导出
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=True,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[导出完成] 已通过 ONNX 结构校验: {onnx_path}")
    return onnx_path


def simplify_onnx_model(
    onnx_path: str,
    optimized_path: str = "optimized_model.onnx",
) -> str:
    """
    使用 onnxslim 进行图级精简（常量折叠、冗余节点消除等）；
    若未安装 onnxslim，则保留原始图并依赖 ONNX Runtime 会话级图优化。
    """
    model = onnx.load(onnx_path)
    try:
        import onnxslim

        slim_model = onnxslim.slim(model)
        onnx.checker.check_model(slim_model)
        onnx.save(slim_model, optimized_path)
        print(f"[图精简完成] 已使用 onnxslim 优化并保存: {optimized_path}")
        return optimized_path
    except ImportError:
        onnx.save(model, optimized_path)
        print(
            "[提示] 未安装 onnxslim（可选安装: pip install onnxslim），"
            f"已复制模型 {optimized_path}，后续将依赖 ONNX Runtime 进行图优化。"
        )
        return optimized_path


def create_session(onnx_path: str, intra_op_threads: int = 4) -> ort.InferenceSession:
    """配置 SessionOptions 与 Execution Provider，创建推理会话。"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options=options, providers=providers)
    print(f"[会话就绪] 实际生效 Providers: {session.get_providers()}")
    return session


def verify_and_benchmark(
    pytorch_model: nn.Module,
    session: ort.InferenceSession,
    in_features: int = 10,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    warmup: int = 20,
    runs: int = 200,
) -> None:
    """校验多 batch 动态维度的数值精度一致性，并统计推理延迟。"""
    pytorch_model.eval()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for bs in batch_sizes:
        x_np = np.random.randn(bs, in_features).astype(np.float32)

        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(x_np)).cpu().numpy()

        ort_out = session.run([output_name], {input_name: x_np})[0]

        max_abs_err = float(np.max(np.abs(pt_out - ort_out)))
        is_close = np.allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)
        assert is_close, f"batch_size={bs} 数值对齐失败，最大绝对误差: {max_abs_err}"

        # Warmup + 计时
        for _ in range(warmup):
            session.run([output_name], {input_name: x_np})

        start = time.perf_counter()
        for _ in range(runs):
            session.run([output_name], {input_name: x_np})
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

        print(
            f"[batch_size={bs:2d}] 对齐通过 | max_abs_err={max_abs_err:.2e} | "
            f"平均单次推理延迟={elapsed_ms:.3f} ms"
        )


def main() -> None:
    in_features = 10
    model = SimpleModel(in_features=in_features, hidden_features=20, out_features=2)

    raw_onnx_path = export_to_onnx(model, onnx_path="simple_model.onnx", in_features=in_features)
    opt_onnx_path = simplify_onnx_model(raw_onnx_path, optimized_path="optimized_model.onnx")
    session = create_session(opt_onnx_path, intra_op_threads=4)
    verify_and_benchmark(model, session, in_features=in_features, batch_sizes=(1, 8, 32))


if __name__ == "__main__":
    main()