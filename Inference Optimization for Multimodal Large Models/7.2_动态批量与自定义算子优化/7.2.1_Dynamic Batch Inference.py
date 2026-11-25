import os
import time
import numpy as np
import onnx
from onnx import TensorProto, helper
import pycuda.autoinit  # noqa: F401 - 初始化 CUDA 主上下文
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def create_sample_onnx_if_missing(
    onnx_path: str = "dynamic_model.onnx",
    in_features: int = 10,
    hidden_features: int = 16,
    out_features: int = 2,
    opset_version: int = 17,
) -> str:
    """若本地不存在 ONNX 模型，则生成一个支持动态 batch 的两层全连接示例模型。"""
    if os.path.exists(onnx_path):
        return onnx_path

    w1 = np.random.randn(in_features, hidden_features).astype(np.float32) * 0.1
    b1 = np.zeros((hidden_features,), dtype=np.float32)
    w2 = np.random.randn(hidden_features, out_features).astype(np.float32) * 0.1
    b2 = np.zeros((out_features,), dtype=np.float32)

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", in_features])
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch_size", out_features])

    w1_init = helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.tobytes(), raw=True)
    b1_init = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.tobytes(), raw=True)
    w2_init = helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.tobytes(), raw=True)
    b2_init = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.tobytes(), raw=True)

    gemm1 = helper.make_node("Gemm", inputs=["input", "w1", "b1"], outputs=["h1"], alpha=1.0, beta=1.0, transB=0)
    relu = helper.make_node("Relu", inputs=["h1"], outputs=["h2"])
    gemm2 = helper.make_node("Gemm", inputs=["h2", "w2", "b2"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)

    graph = helper.make_graph(
        nodes=[gemm1, relu, gemm2],
        name="DynamicMLPGraph",
        inputs=[x_info],
        outputs=[y_info],
        initializer=[w1_init, b1_init, w2_init, b2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"[已生成示例 ONNX] {onnx_path}")
    return onnx_path


def _trt_dtype_to_np(dtype: trt.DataType) -> np.dtype:
    """将 TensorRT 数据类型映射为 NumPy dtype。"""
    mapping = {
        trt.DataType.FLOAT: np.dtype(np.float32),
        trt.DataType.HALF: np.dtype(np.float16),
        trt.DataType.INT8: np.dtype(np.int8),
        trt.DataType.INT32: np.dtype(np.int32),
        trt.DataType.BOOL: np.dtype(np.bool_),
    }
    if hasattr(trt.DataType, "BFLOAT16") and dtype == trt.DataType.BFLOAT16:
        return np.dtype(np.float32)
    if dtype not in mapping:
        raise TypeError(f"不支持映射的 TensorRT 数据类型: {dtype}")
    return mapping[dtype]


def build_dynamic_engine(
    onnx_file_path: str,
    engine_file_path: str,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 32,
    in_features: int = 10,
    workspace_mb: int = 1024,
    enable_fp16: bool = False,
) -> None:
    """使用 BuilderConfig + OptimizationProfile 构建动态 Batch 引擎并序列化保存。"""
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network() as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("解析 ONNX 失败:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

        if enable_fp16 and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)

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
            raise RuntimeError("TensorRT 引擎构建失败，请检查模型算子与日志。")

        with open(engine_file_path, "wb") as f:
            f.write(bytes(serialized_engine))

        print(f"[引擎构建完成] 已保存至 {engine_file_path} (batch 范围: [{min_batch}, {opt_batch}, {max_batch}])")


class PyCudaTrtDynamicRunner:
    """复用 Engine 与 ExecutionContext 执行动态 Batch 推理（适配 TensorRT 10/11）。"""

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
        if not self.input_names or not self.output_names:
            raise RuntimeError("引擎中未找到有效的输入或输出张量。")

        self.stream = cuda.Stream()

    def infer(self, input_data: np.ndarray, measure_kernel_ms: bool = False) -> tuple[np.ndarray, float]:
        """执行单次动态 Batch 推理，返回 (输出数组, 耗时毫秒)。"""
        in_name = self.input_names[0]
        out_name = self.output_names[0]

        in_contig = np.ascontiguousarray(input_data, dtype=np.float32)
        self.context.set_input_shape(in_name, in_contig.shape)

        out_shape = tuple(self.context.get_tensor_shape(out_name))
        if any(dim < 0 for dim in out_shape):
            raise RuntimeError(f"输出维度未完全解析: {out_shape}")

        out_np_dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(out_name))
        output_data = np.empty(out_shape, dtype=out_np_dtype)

        d_input = cuda.mem_alloc(in_contig.nbytes)
        d_output = cuda.mem_alloc(output_data.nbytes)

        try:
            self.context.set_tensor_address(in_name, int(d_input))
            self.context.set_tensor_address(out_name, int(d_output))

            cuda.memcpy_htod_async(d_input, in_contig, self.stream)

            if measure_kernel_ms:
                start_evt = cuda.Event()
                end_evt = cuda.Event()
                start_evt.record(self.stream)
                ok = self.context.execute_async_v3(self.stream.handle)
                end_evt.record(self.stream)
                if not ok:
                    raise RuntimeError("TensorRT 异步推理执行失败。")
                cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
                self.stream.synchronize()
                elapsed_ms = end_evt.time_since(start_evt)
            else:
                t0 = time.perf_counter()
                ok = self.context.execute_async_v3(self.stream.handle)
                if not ok:
                    raise RuntimeError("TensorRT 异步推理执行失败。")
                cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
                self.stream.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return output_data, elapsed_ms
        finally:
            d_input.free()
            d_output.free()


def main() -> None:
    onnx_model_path = "dynamic_model.onnx"
    trt_engine_path = "dynamic_model_trt.engine"
    in_features = 10

    create_sample_onnx_if_missing(onnx_model_path, in_features=in_features, out_features=2)

    if not os.path.exists(trt_engine_path):
        build_dynamic_engine(
            onnx_file_path=onnx_model_path,
            engine_file_path=trt_engine_path,
            min_batch=1,
            opt_batch=8,
            max_batch=32,
            in_features=in_features,
            enable_fp16=False,
        )

    runner = PyCudaTrtDynamicRunner(trt_engine_path)
    batch_sizes = [1, 8, 32]

    for bs in batch_sizes:
        sample = np.random.randn(bs, in_features).astype(np.float32)

        for _ in range(5):
            runner.infer(sample, measure_kernel_ms=False)

        runs = 50
        total_ms = 0.0
        last_out = None
        for _ in range(runs):
            last_out, ms = runner.infer(sample, measure_kernel_ms=False)
            total_ms += ms

        avg_ms = total_ms / runs
        throughput = bs / (avg_ms / 1000.0)
        print(
            f"[batch={bs:2d}] out_shape={last_out.shape} | "
            f"平均耗时={avg_ms:.4f} ms | 吞吐量={throughput:,.1f} samples/s"
        )


if __name__ == "__main__":
    main()