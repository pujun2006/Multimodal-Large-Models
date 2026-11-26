import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_weights_cleanly(model: nn.Module, weight_path: str, device: torch.device) -> nn.Module:
    # 先加载到 CPU 避免峰值显存暴增，再一次性加载完整 state_dict 后搬移到目标设备
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def optimized_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    model.eval()
    results = []
    amp_dtype = torch.bfloat16 if (use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode():  # 比 no_grad 更进一步关闭版本计数等推理无关开销
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(use_amp and device.type == "cuda"),
            ):
                outputs = model(inputs)
            results.append(torch.argmax(outputs, dim=1).cpu())

    return torch.cat(results) if results else torch.empty(0, dtype=torch.long)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleModel()
    torch.save(model.state_dict(), "model_weights.pth")

    model = load_weights_cleanly(model, "model_weights.pth", device)
    predictions = optimized_inference(model, test_loader, device, use_amp=True)
    print(f"推理完成，共处理了 {len(predictions)} 张图片。")