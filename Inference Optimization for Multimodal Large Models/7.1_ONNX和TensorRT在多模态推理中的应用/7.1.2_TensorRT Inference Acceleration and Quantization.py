import os
import time
import numpy as np
import onnx
import tensorrt as trt
import torch
import torch.nn as nn

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class SimpleModel(nn.Module):
    """示例网络：输入 (batch, 10)，输出 (batch, 2)。"""

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
    """导出 PyTorch 模型为 ONNX 并校验结构合法性。"""
    model_cpu = model.eval().cpu()
    dummy_input = torch.randn(1, in_features, dtype=torch.float32)

    try:
        torch.onnx.export(
            model_cpu,
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
    except Exception:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[导出完成] 已生成并通过校验: {onnx_path}")
    return onnx_path


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "BFLOAT16") and dtype == trt.DataType.BFLOAT16:
        return torch.bfloat16
    if hasattr(trt.DataType, "FP8") and dtype == trt.DataType.FP8:
        return torch.float8_e4m3fn
    if dtype not in mapping:
        raise TypeError(f"暂未映射的 TensorRT 数据类型: {dtype}")
    return mapping[dtype]


def build_engine_trt11(
    onnx_file_path: str,
    engine_file_path: str,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 32,
    in_features: int = 10,
    workspace_mb: int = 1024,
) -> bytes:
    """使用 TensorRT 11.x 解析 ONNX、配置动态 batch Profile 并序列化引擎。"""
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network() as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("解析 ONNX 模型失败:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

        input_name = network.get_input(0).name
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_name,
            min=(min_batch, in_features),
            opt=(opt_batch, in_features),
            max=(max_batch, in_features),
        )
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT 11 引擎构建失败，请检查日志输出。")

        engine_bytes = bytes(serialized_engine)
        with open(engine_file_path, "wb") as f:
            f.write(engine_bytes)

        print(f"[构建完成] 已保存引擎至: {engine_file_path}")
        return engine_bytes


class Trt11Runner:
    """封装 TensorRT 11 反序列化、动态 Shape 绑定与异步推理执行。"""

    def __init__(self, engine_file_path: str):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化引擎失败: {engine_file_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("创建 ExecutionContext 失败。")

        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def infer(self, input_np: np.ndarray, stream: torch.cuda.Stream | None = None) -> np.ndarray:
        stream = stream or torch.cuda.current_stream()
        in_name = self.input_names[0]
        out_name = self.output_names[0]

        in_contig = np.ascontiguousarray(input_np, dtype=np.float32)
        self.context.set_input_shape(in_name, in_contig.shape)

        out_shape = tuple(self.context.get_tensor_shape(out_name))
        out_torch_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(out_name))

        with torch.cuda.stream(stream):
            d_input = torch.from_numpy(in_contig).to(device="cuda", non_blocking=True)
            d_output = torch.empty(out_shape, dtype=out_torch_dtype, device="cuda")

            self.context.set_tensor_address(in_name, int(d_input.data_ptr()))
            self.context.set_tensor_address(out_name, int(d_output.data_ptr()))

            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT 11 推理执行失败。")

            out_np = d_output.to(dtype=torch.float32).cpu().numpy()

        stream.synchronize()
        return out_np


def verify_and_benchmark(
    pytorch_model: nn.Module,
    runner: Trt11Runner,
    label: str = "TRT11",
    in_features: int = 10,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    warmup: int = 20,
    runs: int = 200,
) -> None:
    """校验多 batch 输出精度误差并统计单次推理平均延迟。"""
    pytorch_model.eval().cuda()
    stream = torch.cuda.Stream()

    for bs in batch_sizes:
        x_np = np.random.randn(bs, in_features).astype(np.float32)
        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(x_np).cuda()).cpu().numpy()

        trt_out = runner.infer(x_np, stream=stream)
        max_abs_err = float(np.max(np.abs(pt_out - trt_out)))

        for _ in range(warmup):
            runner.infer(x_np, stream=stream)

        start = time.perf_counter()
        for _ in range(runs):
            runner.infer(x_np, stream=stream)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

        print(
            f"[{label} | batch={bs:2d}] shape={trt_out.shape} | "
            f"max_abs_err={max_abs_err:.2e} | 平均延迟={elapsed_ms:.3f} ms"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到可用 CUDA GPU。")

    in_features = 10
    onnx_path = "simple_model.onnx"
    engine_path = "simple_model_trt11.engine"

    pt_model = SimpleModel(in_features=in_features, hidden_features=20, out_features=2)

    if not os.path.exists(onnx_path):
        export_to_onnx(pt_model, onnx_path=onnx_path, in_features=in_features)

    build_engine_trt11(onnx_path, engine_path, in_features=in_features)

    runner = Trt11Runner(engine_path)
    verify_and_benchmark(pt_model, runner, label="TRT11", in_features=in_features)


if __name__ == "__main__":
    main()
