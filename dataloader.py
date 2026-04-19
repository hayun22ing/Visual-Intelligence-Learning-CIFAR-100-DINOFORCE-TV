"""
dataloader.py
CIFAR-100 DataLoader + SUPERCLASS_MAP 자동 추출 + aug 종류별 transform
"""
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


# ── SUPERCLASS_MAP 데이터셋에서 직접 추출 ────────────────────────────────────

def build_superclass_map(data_root='./data'):
    """
    CIFAR-100 메타데이터에서 fine→coarse 매핑 자동 추출
    하드코딩 없이 공식 정의 그대로 사용
    """
    path = f'{data_root}/cifar-100-python/train'
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')

    fine   = np.array(d[b'fine_labels'])
    coarse = np.array(d[b'coarse_labels'])

    sc_map = np.zeros(100, dtype=int)
    for fi, ci in zip(fine, coarse):
        sc_map[fi] = ci

    return torch.tensor(sc_map.tolist())   # shape (100,)


# ── 증강 종류 ─────────────────────────────────────────────────────────────────

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_transform(aug='basic', mode='train'):
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if mode == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if aug == 'basic':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if aug == 'cutout':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes=1, length=16),
        ])

    if aug == 'strong':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes=1, length=16),
        ])

    raise ValueError(f'알 수 없는 aug 종류: {aug}')


class Cutout:
    """랜덤 정사각형 영역을 0으로 마스킹"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        h, w = img.size(-2), img.size(-1)
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0
        return img * mask


# ── CIFAR100 with coarse label ────────────────────────────────────────────────

class CIFAR100WithCoarse(Dataset):
    """
    torchvision CIFAR100를 감싸서 (image, fine_label, coarse_label) 반환
    """
    def __init__(self, base_dataset, sc_map):
        self.base   = base_dataset
        self.sc_map = sc_map   # (100,) tensor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, fine = self.base[idx]
        coarse    = self.sc_map[fine].item()
        return img, fine, coarse


# ── DataLoader 생성 ───────────────────────────────────────────────────────────

def get_dataloaders(data_root='./data', aug='basic',
                    batch_size=64, num_workers=4):
    """
    train(50k) / val(10k) DataLoader 반환
    - train: CIFAR-100 공식 train set 전체 (50,000장)
    - val  : CIFAR-100 공식 test set 전체 (10,000장)

    Returns:
        train_loader, val_loader, sc_map
    """
    train_transform = get_transform(aug=aug,   mode='train')
    test_transform  = get_transform(aug='basic', mode='test')

    # 다운로드 먼저 (build_superclass_map이 pickle 파일을 직접 읽으므로)
    train_base = datasets.CIFAR100(data_root, train=True,
                                   download=True, transform=train_transform)
    val_base   = datasets.CIFAR100(data_root, train=False,
                                   download=True, transform=test_transform)

    # SUPERCLASS_MAP 추출 (다운로드 완료 후)
    sc_map = build_superclass_map(data_root)

    # coarse label 래핑
    train_ds = CIFAR100WithCoarse(train_base, sc_map)
    val_ds   = CIFAR100WithCoarse(val_base,   sc_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f'[Data] Train {len(train_ds):,} / Val {len(val_ds):,} '
          f'| aug={aug} | batch={batch_size}')
    return train_loader, val_loader, sc_map
