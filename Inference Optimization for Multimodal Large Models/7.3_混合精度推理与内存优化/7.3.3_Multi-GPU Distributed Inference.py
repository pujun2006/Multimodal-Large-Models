import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class IndexedDataset(Dataset):
    """包装数据集，同时返回样本与原始全局索引，便于分布式结果去重和还原顺序。"""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return x, y, idx


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def setup(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def prepare_dataset(rank: int, root: str = "./data") -> Dataset:
    # CIFAR-10 标准化使用标准三通道均值与方差
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # 仅由 rank 0 负责下载，其他进程等待同步完成后再读取本地缓存
    if rank == 0:
        base_ds = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
    torch.distributed.barrier()

    if rank != 0:
        base_ds = torchvision.datasets.CIFAR10(
            root=root, train=False, download=False, transform=transform
        )
    torch.distributed.barrier()

    return IndexedDataset(base_ds)


def run_inference(rank: int, world_size: int, batch_size: int = 64, num_workers: int = 2):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    try:
        dataset = prepare_dataset(rank, root="./data")
        original_len = len(dataset)

        # 推理阶段关闭 shuffle；drop_last=False 保证不丢弃尾部样本
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        model = SimpleModel(num_classes=10).to(device)
        # 若有已训练权重，在此加载：
        # state_dict = torch.load("checkpoint.pt", map_location=device)
        # model.load_state_dict(state_dict)
        model.eval()

        local_preds = []
        local_indices = []
        local_raw_batches = 0

        with torch.inference_mode():
            for inputs, _, indices in dataloader:
                local_raw_batches += inputs.size(0)
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().tolist()

                local_preds.extend(preds)
                local_indices.extend(indices.tolist())

        # 将各 rank 的 (索引, 预测结果) 收集到全局列表
        gathered_pairs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gathered_pairs, list(zip(local_indices, local_preds))
        )

        if rank == 0:
            # 按原始索引去重（消除 DistributedSampler 为对齐长度而产生的 padding）
            merged = {}
            for rank_pairs in gathered_pairs:
                for idx, pred in rank_pairs:
                    if idx < original_len and idx not in merged:
                        merged[idx] = pred

            ordered_preds = [merged[i] for i in range(original_len)]
            print(
                f"推理完成：原始样本数={original_len}，去重后有效预测数={len(ordered_preds)}，"
                f"Rank 0 含 padding 的遍历样本数={local_raw_batches}"
            )
            print(f"前 10 条预测类别：{ordered_preds[:10]}")

    finally:
        cleanup()


def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("未检测到可用 GPU，无法运行 NCCL 多 GPU 推理。")

    mp.spawn(
        run_inference,
        args=(world_size, 64, 2),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()