import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 归一化参数与张量合法上下界（像素 [0, 1] 经 mean=0.5, std=0.5 映射后为 [-1.0, 1.0]）
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
MIN_NORM = -1.0
MAX_NORM = 1.0


def to_norm_eps(eps_pixel: float, std: list[float] = STD, device: torch.device | None = None) -> torch.Tensor:
    """将 [0, 1] 像素空间扰动半径转换为按通道归一化的张量半径。"""
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return eps_pixel / std_tensor


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    eps_pixel: float = 0.03,
) -> torch.Tensor:
    """单步 FGSM 对抗样本生成（输入已处于归一化空间）。"""
    was_training = model.training
    model.eval()

    eps_tensor = to_norm_eps(eps_pixel, STD, inputs.device)
    x_adv = inputs.detach().clone().requires_grad_(True)

    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps_tensor * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, MIN_NORM, MAX_NORM)

    if was_training:
        model.train()
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    eps_pixel: float = 0.03,
    alpha_pixel: float = 0.0075,
    steps: int = 7,
    random_start: bool = True,
) -> torch.Tensor:
    """多步 PGD（L_inf）对抗样本生成，用于更可靠地评估鲁棒性。"""
    was_training = model.training
    model.eval()

    eps_tensor = to_norm_eps(eps_pixel, STD, inputs.device)
    alpha_tensor = to_norm_eps(alpha_pixel, STD, inputs.device)
    x_orig = inputs.detach()

    if random_start:
        rand_noise = torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps_tensor
        x_adv = torch.clamp(x_orig + rand_noise, MIN_NORM, MAX_NORM)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha_tensor * x_adv.grad.sign()
            # 投影回原始输入周围的 L_inf 扰动球内，并裁剪到合法归一化区间
            delta = torch.clamp(x_adv - x_orig, min=-eps_tensor, max=eps_tensor)
            x_adv = torch.clamp(x_orig + delta, MIN_NORM, MAX_NORM)

    if was_training:
        model.train()
    return x_adv.detach()


def add_gaussian_noise(inputs: torch.Tensor, sigma_pixel: float = 0.05) -> torch.Tensor:
    """在归一化张量上叠加高斯噪声，测试自然扰动下的稳定性。"""
    sigma_tensor = to_norm_eps(sigma_pixel, STD, inputs.device)
    noisy = inputs.detach() + torch.randn_like(inputs) * sigma_tensor
    return torch.clamp(noisy, MIN_NORM, MAX_NORM)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    use_adv_train: bool = False,
    eps_pixel: float = 0.03,
    alpha_pixel: float = 0.0075,
    pgd_steps: int = 5,
) -> None:
    """训练单个 epoch；当 use_adv_train=True 时使用 PGD 对抗样本更新参数。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        train_inputs = (
            pgd_attack(
                model,
                inputs,
                labels,
                eps_pixel=eps_pixel,
                alpha_pixel=alpha_pixel,
                steps=pgd_steps,
                random_start=True,
            )
            if use_adv_train
            else inputs
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(train_inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    mode_tag = "ADV-PGD" if use_adv_train else "CLEAN"
    print(
        f"Epoch {epoch:02d} [{mode_tag}] | "
        f"Train Loss: {running_loss / total:.4f} | "
        f"Train Acc: {100.0 * correct / total:.2f}%"
    )


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    eps_pixel: float = 0.03,
    noise_sigma_pixel: float = 0.05,
) -> dict[str, float]:
    """同时评估干净样本、高斯噪声样本、FGSM 与 PGD 对抗样本准确率。"""
    model.eval()
    clean_correct = 0
    noise_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = labels.size(0)

        with torch.no_grad():
            clean_preds = model(inputs).argmax(dim=1)
            clean_correct += clean_preds.eq(labels).sum().item()

            noisy_inputs = add_gaussian_noise(inputs, sigma_pixel=noise_sigma_pixel)
            noise_preds = model(noisy_inputs).argmax(dim=1)
            noise_correct += noise_preds.eq(labels).sum().item()

        fgsm_inputs = fgsm_attack(model, inputs, labels, eps_pixel=eps_pixel)
        pgd_inputs = pgd_attack(
            model,
            inputs,
            labels,
            eps_pixel=eps_pixel,
            alpha_pixel=eps_pixel / 4.0,
            steps=10,
            random_start=True,
        )

        with torch.no_grad():
            fgsm_correct += model(fgsm_inputs).argmax(dim=1).eq(labels).sum().item()
            pgd_correct += model(pgd_inputs).argmax(dim=1).eq(labels).sum().item()

        total += bs

    metrics = {
        "clean_acc": 100.0 * clean_correct / total,
        "noise_acc": 100.0 * noise_correct / total,
        "fgsm_acc": 100.0 * fgsm_correct / total,
        "pgd_acc": 100.0 * pgd_correct / total,
    }
    print(
        f"  Eval -> Clean: {metrics['clean_acc']:.2f}% | "
        f"Noise(sigma={noise_sigma_pixel}): {metrics['noise_acc']:.2f}% | "
        f"FGSM(eps={eps_pixel}): {metrics['fgsm_acc']:.2f}% | "
        f"PGD(eps={eps_pixel}): {metrics['pgd_acc']:.2f}%"
    )
    return metrics


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_mem)

    model = models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 前半程标准训练建立基础特征，后半程开启 PGD 对抗训练；实际实验中可另起独立基线对比完整训练曲线
    adv_start_epoch = 11

    for epoch in range(1, epochs + 1):
        use_adv = epoch >= adv_start_epoch
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            use_adv_train=use_adv,
            eps_pixel=0.03,
            alpha_pixel=0.0075,
            pgd_steps=5,
        )
        scheduler.step()
        evaluate(model, test_loader, device=device, eps_pixel=0.03, noise_sigma_pixel=0.05)


if __name__ == "__main__":
    main()