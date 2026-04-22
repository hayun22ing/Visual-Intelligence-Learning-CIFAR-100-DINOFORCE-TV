"""
model.py
WideResNet-28-10 + AUX head + Embedding(SupCon) 지원
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── WideResNet ────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.drop  = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.drop(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet-28-10 for CIFAR-100

    method:
      'base' → fc(100)만 반환
      'aux'  → fc(100), fc_super(20) 반환
      'emb'  → fc(100), fc_super(20), L2 정규화 embedding 반환
    """
    def __init__(self, depth=28, widen=10, num_classes=100,
                 num_super=20, dropout=0.3, method='base'):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        ch = [16, 16 * widen, 32 * widen, 64 * widen]
        self.method = method

        self.conv1  = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlock, n, ch[0], ch[1], 1, dropout)
        self.layer2 = self._make_layer(BasicBlock, n, ch[1], ch[2], 2, dropout)
        self.layer3 = self._make_layer(BasicBlock, n, ch[2], ch[3], 2, dropout)
        self.bn     = nn.BatchNorm2d(ch[3])
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(ch[3], num_classes)

        if method in ('aux', 'emb'):
            self.fc_super = nn.Linear(ch[3], num_super)

        self._init_weights()

    def _make_layer(self, block, n, in_ch, out_ch, stride, dropout):
        layers = [block(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n):
            layers.append(block(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = self.pool(out)
        feat = out.view(out.size(0), -1)   # (B, 640)

        logits_fine = self.fc(feat)

        if self.method == 'base':
            return logits_fine

        logits_super = self.fc_super(feat)

        if self.method == 'aux':
            return logits_fine, logits_super

        # emb: L2 정규화 embedding 추가 반환
        emb = F.normalize(feat, dim=1)
        return logits_fine, logits_super, emb


# ── Supervised Contrastive Loss (슈퍼클래스 단위) ─────────────────────────────

class SuperclassSupConLoss(nn.Module):
    """
    같은 슈퍼클래스 샘플들을 embedding 공간에서 뭉치게 만드는 손실
    temperature=0.07 권장
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, emb, coarse_labels):
        """
        emb          : L2 정규화된 embedding (B, D)
        coarse_labels: 슈퍼클래스 레이블 (B,)
        """
        B = emb.size(0)
        # emb는 모델에서 이미 L2 정규화되어 입력됨 (중복 정규화 제거)
        sim = torch.matmul(emb, emb.T) / self.temp      # (B, B)

        labels = coarse_labels.unsqueeze(1)
        mask   = (labels == labels.T).float()            # 같은 SC 마스크
        mask.fill_diagonal_(0)                           # 자기 자신 제외

        # logsumexp 트릭으로 수치 안정성 확보
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max)
        exp_sim = exp_sim * (1 - torch.eye(B, device=emb.device))  # 대각 제거

        log_prob  = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = mask.sum(dim=1)
        valid     = pos_count > 0

        if not valid.any():
            return emb.sum() * 0.0  # computation graph 유지, gradient 단절 방지

        loss = -(mask * log_prob).sum(dim=1)
        loss = loss[valid] / pos_count[valid]
        return loss.mean()


# ── SAM Optimizer ─────────────────────────────────────────────────────────────

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021)"""
    def __init__(self, params, base_optimizer, rho=0.05,
                 adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) \
                      * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad)
            .norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
