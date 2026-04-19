"""
trainer.py 수정 부분
- Mixup / CutMix 적용 확률을 1.0(100%)으로 고정
"""
import torch
import numpy as np

def rand_bbox(size, lam):
    """CutMix를 위한 바운딩 박스 생성 함수 (기존과 동일)"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_mix_aug(imgs, targets, mix_aug='both', alpha=1.0):
    """
    [수정됨] 확률 체크(mix_prob)를 아예 없애고 무조건(100%) 증강을 적용합니다.
    """
    if mix_aug is None or mix_aug == 'none':
        return imgs, targets, targets, 1.0

    # 'both'일 경우 50:50 확률로 Mixup과 CutMix 중 하나를 랜덤 선택 (건너뛰는 경우는 없음!)
    if mix_aug == 'both':
        mix_type = 'mixup' if np.random.rand() < 0.5 else 'cutmix'
    else:
        mix_type = mix_aug

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(imgs.size(0)).to(imgs.device)
    
    target_a = targets
    target_b = targets[rand_index]

    if mix_type == 'mixup':
        imgs = lam * imgs + (1 - lam) * imgs[rand_index]
    elif mix_type == 'cutmix':
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # 픽셀이 섞인 실제 비율로 lambda 재조정
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

    return imgs, target_a, target_b, lam