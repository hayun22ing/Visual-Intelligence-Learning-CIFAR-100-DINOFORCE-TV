"""
trainer.py
train_one_epoch / evaluate — BASE / AUX / EMB 공통 지원
mix_aug: None / 'mixup' / 'cutmix' / 'both' (랜덤 선택)
"""
import torch
import torch.nn as nn
import numpy as np
from model import SuperclassSupConLoss


supcon_loss_fn = None   # 전역 캐시 (한 번만 생성)


def _get_supcon(temperature=0.07, device='cpu'):
    global supcon_loss_fn
    if supcon_loss_fn is None:
        supcon_loss_fn = SuperclassSupConLoss(temperature=temperature).to(device)
    return supcon_loss_fn


def compute_superclass_accuracy(logits, fine_targets, sc_map):
    """
    Top-5 예측 중 정답과 같은 슈퍼클래스에 속하는 비율 평균
    logits      : (B, 100)
    fine_targets: (B,)
    sc_map      : (100,) tensor
    """
    sc_map   = sc_map.to(logits.device)
    top5     = logits.topk(5, dim=1).indices          # (B, 5)
    gt_sc    = sc_map[fine_targets].unsqueeze(1)       # (B, 1)
    pred_sc  = sc_map[top5]                            # (B, 5)
    match    = (pred_sc == gt_sc).float().sum(dim=1)   # (B,)
    return (match / 5.0).mean().item()


# ── Mixup / CutMix ────────────────────────────────────────────────────────────

def _mixup(imgs, fine_t, coarse_t, alpha=0.2):
    """
    반환: mixed_imgs, fine_a, fine_b, coarse_a, coarse_b, lam
    CE loss  : lam * CE(logits, fine_a)  + (1-lam) * CE(logits, fine_b)
    SupCon   : coarse_a 사용 (원본 레이블 유지)
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = imgs.size(0)
    idx = torch.randperm(B, device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, fine_t, fine_t[idx], coarse_t, coarse_t[idx], lam


def _cutmix(imgs, fine_t, coarse_t, alpha=1.0):
    """
    반환: mixed_imgs, fine_a, fine_b, coarse_a, coarse_b, lam
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = imgs.shape
    idx = torch.randperm(B, device=imgs.device)

    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)   # 실제 lam 재계산
    return mixed, fine_t, fine_t[idx], coarse_t, coarse_t[idx], lam


def _apply_mix(imgs, fine_t, coarse_t, mix_aug, mix_alpha):
    """mix_aug에 따라 mixup / cutmix / both(랜덤) 적용"""
    if mix_aug == 'both':
        mix_aug = 'mixup' if np.random.rand() < 0.5 else 'cutmix'
    if mix_aug == 'mixup':
        return _mixup(imgs, fine_t, coarse_t, mix_alpha)
    if mix_aug == 'cutmix':
        return _cutmix(imgs, fine_t, coarse_t, mix_alpha)
    return imgs, fine_t, None, coarse_t, None, 1.0  # mix 없음


# ── Forward + Loss ────────────────────────────────────────────────────────────

def _mixed_ce(criterion, logits, fine_a, fine_b, lam):
    """Mixup/CutMix용 CE loss"""
    return lam * criterion(logits, fine_a) + (1 - lam) * criterion(logits, fine_b)


def _forward_and_loss(model, imgs, fine_t, coarse_t,
                      criterion, method, alpha, beta, temperature,
                      loss_rule='single',
                      fine_t_b=None, coarse_t_b=None, lam=1.0):
    """
    method별 forward + loss 계산
    fine_t_b / coarse_t_b / lam : Mixup·CutMix 적용 시 전달
    """
    use_mix = (fine_t_b is not None) and (lam < 1.0)

    if method == 'base':
        logits = model(imgs)
        loss   = _mixed_ce(criterion, logits, fine_t, fine_t_b, lam) \
                 if use_mix else criterion(logits, fine_t)
        return loss, logits

    if method == 'aux':
        logits_fine, logits_super = model(imgs)
        l_fine  = _mixed_ce(criterion, logits_fine, fine_t, fine_t_b, lam) \
                  if use_mix else criterion(logits_fine, fine_t)
        # super head는 원본 레이블 사용 (mix 미적용)
        l_super = criterion(logits_super, coarse_t)
        if loss_rule == 'normalized':
            loss = (l_fine + alpha * l_super) / (1.0 + alpha)
        else:
            loss = l_fine + alpha * l_super
        return loss, logits_fine

    if method == 'emb':
        logits_fine, _, emb = model(imgs)
        l_fine   = _mixed_ce(criterion, logits_fine, fine_t, fine_t_b, lam) \
                   if use_mix else criterion(logits_fine, fine_t)
        # SupCon은 원본 coarse 레이블 사용 (mix 미적용)
        supcon   = _get_supcon(temperature, imgs.device)
        l_supcon = supcon(emb, coarse_t)
        if loss_rule == 'normalized':
            loss = (l_fine + beta * l_supcon) / (1.0 + beta)
        else:
            loss = l_fine + beta * l_supcon
        return loss, logits_fine

    raise ValueError(f'알 수 없는 method: {method}')


def train_one_epoch(model, loader, optimizer, criterion, device,
                    method='base', opt_type='sgd',
                    alpha=0.2, beta=0.05, temperature=0.07,
                    loss_rule='single',
                    mix_aug=None, mix_alpha=0.2):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, fine_t, coarse_t in loader:
        imgs, fine_t, coarse_t = (imgs.to(device),
                                   fine_t.to(device),
                                   coarse_t.to(device))

        # Mixup / CutMix 적용
        if mix_aug:
            imgs, fine_t, fine_t_b, coarse_t, coarse_t_b, lam = \
                _apply_mix(imgs, fine_t, coarse_t, mix_aug, mix_alpha)
        else:
            fine_t_b = coarse_t_b = None
            lam = 1.0

        mix_kwargs = dict(fine_t_b=fine_t_b, coarse_t_b=coarse_t_b, lam=lam)

        if opt_type == 'sam':
            loss, logits = _forward_and_loss(
                model, imgs, fine_t, coarse_t,
                criterion, method, alpha, beta, temperature, loss_rule,
                **mix_kwargs)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            loss2, _ = _forward_and_loss(
                model, imgs, fine_t, coarse_t,
                criterion, method, alpha, beta, temperature, loss_rule,
                **mix_kwargs)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            log_loss = loss2

        else:
            optimizer.zero_grad()
            loss, logits = _forward_and_loss(
                model, imgs, fine_t, coarse_t,
                criterion, method, alpha, beta, temperature, loss_rule,
                **mix_kwargs)
            loss.backward()
            optimizer.step()
            log_loss = loss

        total_loss += log_loss.item()
        _, pred = logits.max(1)
        total   += fine_t.size(0)
        correct += pred.eq(fine_t).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, sc_map, method='base'):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    sc_sum     = 0.0

    for imgs, fine_t, coarse_t in loader:
        imgs, fine_t, coarse_t = (imgs.to(device),
                                   fine_t.to(device),
                                   coarse_t.to(device))

        if method == 'base':
            logits = model(imgs)
        elif method == 'aux':
            logits, _ = model(imgs)
        else:
            logits, _, _ = model(imgs)

        loss = criterion(logits, fine_t)

        total_loss += loss.item()
        _, pred = logits.max(1)
        total   += fine_t.size(0)
        correct += pred.eq(fine_t).sum().item()
        sc_sum  += compute_superclass_accuracy(logits, fine_t, sc_map) * fine_t.size(0)

    n = len(loader)
    return (total_loss / n,
            100.0 * correct / total,
            100.0 * sc_sum / total)
