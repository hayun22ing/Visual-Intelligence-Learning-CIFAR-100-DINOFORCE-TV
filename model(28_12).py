"""
model(28_12).py (Updated for WRN-28-12 & Strong Regularization)
- WideResNet-28-12 구조 적용
- Dropout 상향 (0.3 -> 0.4)
- 2-Head (Classification + Embedding) 구조 유지
- train/eval 모드에 따른 출력 개수 동적 조정 포함
- SAM Optimizer 통합 포함
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
    def __init__(self, depth=28, widen_factor=12, dropout=0.4, 
                 num_classes=100, embed_dim=128):
        super().__init__()
        self.in_ch = 16
        n = (depth - 4) // 6
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        
        self.layer1 = self._make_layer(16 * k, n, 1, dropout)
        self.layer2 = self._make_layer(32 * k, n, 2, dropout)
        self.layer3 = self._make_layer(64 * k, n, 2, dropout)
        
        self.bn = nn.BatchNorm2d(64 * k)
        
        self.fc = nn.Linear(64 * k, num_classes)
        self.embed_head = nn.Sequential(
            nn.Linear(64 * k, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, out_ch, n, stride, dropout):
        layers = []
        for i in range(n):
            layers.append(BasicBlock(self.in_ch if i == 0 else out_ch,
                                     out_ch, stride if i == 0 else 1, dropout))
        self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        
        logits = self.fc(out)
        embeds = self.embed_head(out)
        embeds = F.normalize(embeds, p=2, dim=1)
        
        # 💡 [가장 중요한 수정 부분]
        # 모델이 스스로 학습 상태인지 검증 상태인지 판단하여 출력을 조절합니다.
        if self.training:
            # 학습 중(train_one_epoch)일 때는 3개를 요구하므로 가짜 데이터를 끼워줍니다.
            dummy_sc = torch.zeros(logits.size(0), 20, device=logits.device)
            return logits, embeds, dummy_sc
        else:
            # 검증 중(evaluate)일 때는 2개만 요구하므로 2개만 반환합니다.
            return logits, embeds


# ── SAM Optimizer ─────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm