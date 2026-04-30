import os
import argparse
import torch
import importlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 괄호가 포함된 파일명 안전하게 임포트
dataloader = importlib.import_module("dataloader")
model_module = importlib.import_module("model(28_12)")

def evaluate_best_model(checkpoint_path='./checkpoints/dinoforce_final_best_seed100.pth', data_root='./data'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Evaluation Started on {device}...")

    # 1. Test Dataset Load (공식 10,000장)
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    # dataloader.py의 매핑 함수 활용
    sc_map = dataloader.build_superclass_map(data_root)
    test_base = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
    test_ds = dataloader.CIFAR100WithCoarse(test_base, sc_map)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    print(f"✅ CIFAR-100 Test Dataset Loaded: {len(test_ds)} images")

    # 2. Model Load
    model = model_module.WideResNet(depth=28, widen_factor=12, dropout=0.0).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please check the path.")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"✅ Checkpoint Loaded (Epoch: {ckpt['epoch']})")

    # 3. Evaluation
    model.eval()
    val_top1, sc_score, total = 0, 0, 0
    fine_to_super_tensor = sc_map.to(device)

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out_fine, _ = model(inputs)
            
            val_top1 += out_fine.max(1)[1].eq(targets).sum().item()
            total += targets.size(0)
            
            # SuperClass 계산
            pred_sc = fine_to_super_tensor[out_fine.topk(5, 1)[1]]
            gt_sc = fine_to_super_tensor[targets].view(-1, 1)
            sc_score += (pred_sc == gt_sc).sum(dim=1).float().mean().item() * targets.size(0)

    test_acc = 100. * val_top1 / total
    test_sc = 100. * (sc_score / 5.0) / total

    print("="*50)
    print("🎯 [Final Evaluation Results]")
    print(f"▶ Top-1 Accuracy: {test_acc:.2f}%")
    print(f"▶ Super-Class Accuracy: {test_sc:.2f}%")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/dinoforce_final_best_seed100.pth')
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()
    evaluate_best_model(checkpoint_path=args.checkpoint, data_root=args.data_root)