import ctypes
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PLUGIN_LIBRARY = "custom_plugin.so"
ctypes.CDLL(PLUGIN_LIBRARY)


def get_custom_plugin(scale: float):
  registry = trt.get_plugin_registry()
  creator = registry.get_plugin_creator("CustomScalePlugin", "1", "")
  if creator is None:
    for c in registry.plugin_creator_list:
      if c.name == "CustomScalePlugin":
        creator = c
        break
  if creator is None:
    raise RuntimeError("未找到 CustomScalePlugin 插件创建器")

  fields = [
      trt.PluginField(
          "scale", np.array([scale], dtype=np.float32), trt.PluginFieldType.FLOAT32
      ),
  ]
  return creator.create_plugin(
      name="custom_scale", field_collection=trt.PluginFieldCollection(fields)
  )


def build_engine(min_bs=1, opt_bs=8, max_bs=16, scale=2.0):
  explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  with (
      trt.Builder(TRT_LOGGER) as builder,
      builder.create_network(explicit_batch) as network,
      builder.create_builder_config() as config,
  ):

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
      config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.add_input(
        name="input", dtype=trt.float32, shape=(-1, 10)
    )
    plugin = get_custom_plugin(scale=scale)
    layer = network.add_plugin_v2(inputs=[input_tensor], plugin=plugin)

    output_tensor = layer.get_output(0)
    output_tensor.name = "output"
    network.mark_output(output_tensor)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(min_bs, 10),
        opt=(opt_bs, 10),
        max=(max_bs, 10),
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
      raise RuntimeError("TensorRT 引擎序列化构建失败")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)


def run_inference(engine, input_data: np.ndarray):
  stream = cuda.Stream()
  with engine.create_execution_context() as context:
    context.set_input_shape("input", input_data.shape)
    output_shape = tuple(context.get_tensor_shape("output"))
    output_data = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))

    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    d_input.free()
    d_output.free()
    return output_data


if __name__ == "__main__":
  engine = build_engine(min_bs=1, opt_bs=8, max_bs=16, scale=2.0)
  x = np.random.rand(8, 10).astype(np.float32)
  y = run_inference(engine, x)
  print("输出形状:", y.shape)
  print("最大误差:", np.max(np.abs(y - 2.0 * x)))