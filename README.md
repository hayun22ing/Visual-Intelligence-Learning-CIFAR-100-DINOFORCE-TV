# CIFAR-100 Classification Challenge: Team DINOFORCE TV

This project aims to develop a high-performance classifier on the CIFAR-100 dataset by optimizing both fine-grained class prediction and superclass-level consistency. The following document describes the model architecture, training strategies, and reproducibility setup in detail according to the submission guidelines.

---

## 1. Project Overview

- **Dataset**: CIFAR-100 (50,000 training / 10,000 test images)
- **Constraint**:
  - Training from scratch (no pretrained weights)
  - No external data usage
  - Single GPU environment

- **Metric**:
  - Top-1 Accuracy
  - Super-Class (SC) Accuracy (based on Top-5 prediction density)

- **Goal**:
  - Reduce fine-grained confusion between similar classes
  - Improve semantic consistency at the superclass level

---

## 2. Requirements (Dependencies)

The following environment is required to run this project:

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.4.0
wandb>=0.15.0 (optional, for experiment logging)
```

---

## 3. Model Architecture: WideResNet-28-12

- **Backbone Capacity**
  Compared to the standard WideResNet-28-10, we increased the widen factor to 12 (WRN-28-12), resulting in approximately 1.44× more parameters and enhanced representation power. This enables better discrimination of subtle visual differences.

- **Strategic Dropout Removal**
  Since strong data augmentation (Mixup/CutMix) and hierarchical loss already provide sufficient regularization, dropout was removed (set to 0.0) to avoid limiting model capacity.

---

## 4. Key Strategies & Techniques

### 4.1 Hierarchical Loss Design

- **Hierarchical Label Smoothing**
  Instead of a one-hot target, we use a soft target distribution:
  - 85% probability assigned to the ground-truth class
  - The remaining 15% is distributed among sibling classes within the same superclass (80% of the residual mass)

- **Hybrid Objective**
  Based on the idea that “not all mistakes are equally wrong,” the model is trained to:
  - Accurately classify fine classes
  - Maintain consistency within superclass structures
    This significantly improves SC Accuracy.

---

### 4.2 Optimization & Training

- **SAM (Sharpness-Aware Minimization)**
  SAM is applied to find flatter minima, improving generalization performance.

- **Training Schedule**
  - Batch Size: 256
  - Initial Learning Rate: 0.2
  - Epochs: 300
  - Scheduler: Cosine Annealing

This setup enables stable convergence and high final performance.

---

### 4.3 Advanced Augmentation

- **High-Intensity Mix-Augmentation**
  Mixup and CutMix are applied alternately with 100% probability per batch.
  This significantly improves robustness and prevents overfitting.

---

## 5. Final Performance

Results from three independent runs with different random seeds (300 epochs each):

| Model / Seed         | Top-1 Accuracy    | Super-Class (SC) Accuracy | Notes     |
| -------------------- | ----------------- | ------------------------- | --------- |
| WRN-28-12 (Seed 42)  | 85.04%            | 89.46%                    | Confirmed |
| WRN-28-12 (Seed 77)  | 85.37%            | 89.82%                    | Confirmed |
| WRN-28-12 (Seed 100) | **85.73%**        | **89.76%**                | Best Run  |
| **Final Mean ± Std** | **85.38 ± 0.35%** | **89.68 ± 0.19%**         |           |

---

## 6. How to Train & Evaluate

### Data Split Clarification

- **Training**: 50,000 CIFAR-100 `train=True` images were strictly used for model updating.
- **Evaluation**: The official 10,000 CIFAR-100 `train=False` test images were strictly isolated for evaluation only.

### 6.1 Training

The project is designed to run end-to-end within a Jupyter Notebook (`run_all_wrn_28_12_2.ipynb`).

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook in Google Colab or a local Jupyter environment.
3. Run the initial cells to load required libraries and `.py` modules.
4. Execute the main training cell (`run_experiment`).
5. Training, validation, and evaluation will run automatically.
   - Metrics (Top-1 Acc, SC Acc) are logged every epoch
   - Logs are saved to `log_wrn28_12_scratch.json`
   - Best checkpoints are automatically saved in the `checkpoints/` directory

### 6.2 Evaluation (Testing the Best Checkpoint)

To quickly evaluate the trained model on the 10,000 Test Set without running the Jupyter Notebook, use the standalone evaluation script.

```bash
python evaluate.py
```

This script will load the official 10,000 test images, load the best `.pth` checkpoint, and output the exact Top-1 and Super-Class Accuracy.

---

## 7. Random Seed Configuration

To ensure full reproducibility, all random generators (`numpy`, `torch`, `cuda`) are controlled.

- Official seeds used: **42, 77, 100**

### Changing the Seed

Modify the `seed` value in the configuration dictionary:

```python
_final_scratch_config = {
    'run_id': 303,
    'memo': 'SCRATCH: WRN-28-12, BS256, Drop 0.0, Hierarchical LS',
    'seed': 77,

    'depth': 28,
    'widen_factor': 12,
}
```

Then run:

```python
run_experiment(_final_scratch_config, device)
```

---

## 8. File Structure

```text
├── dataloader.py             # CIFAR-100 loading, preprocessing, superclass mapping
├── evaluate.py               # Standalone evaluation script for the 10k Test Set
├── model(28_12).py          # WRN-28-12 architecture + SAM optimizer
├── trainer(28_12).py        # Training loop, validation, hierarchical loss
├── run_all_wrn_28_12_2.ipynb # Experiment orchestration and config
└── README.md                # Documentation
```

---

# CIFAR-100 Classification Challenge: Team DINOFORCE TV (한글)

본 프로젝트는 CIFAR-100 데이터셋을 활용하여 세부 클래스(Fine-class) 및 슈퍼클래스(Super-class) 분류 성능을 최적화하는 모델을 개발하는 것을 목표로 합니다. 제출 가이드라인에 맞추어 모델 아키텍처, 학습 전략, 재현성 검증 방법 등을 상세히 기술합니다.

---

## 1. Project Overview

- **Dataset**: CIFAR-100 (50,000 Training / 10,000 Test images)
- **Constraint**: 'From Scratch' 학습 (사전 학습된 가중치 사용 금지), 외부 데이터 사용 금지, 단일 GPU 환경
- **Metric**: Top-1 Accuracy & Super-Class (SC) Accuracy (Top-5 기준)
- **Goal**: 세부 클래스 간 혼동을 줄이고 슈퍼클래스 구조 이해 향상

---

## 2. Requirements (Dependencies)

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.4.0
wandb>=0.15.0
```

---

## 3. Model Architecture

WideResNet-28-12를 사용하여 모델의 표현력을 확장하고, Dropout을 제거하여 성능 저하를 방지했습니다.

---

## 4. Key Strategies

- Hierarchical Label Smoothing
- SAM Optimizer
- Mixup / CutMix

---

## 5. Final Performance

| Model / Seed         | Top-1 Accuracy    | Super-Class (SC) Accuracy | Notes     |
| -------------------- | ----------------- | ------------------------- | --------- |
| WRN-28-12 (Seed 42)  | 85.04%            | 89.46%                    | Confirmed |
| WRN-28-12 (Seed 77)  | 85.37%            | 89.82%                    | Confirmed |
| WRN-28-12 (Seed 100) | **85.73%**        | **89.76%**                | Best Run  |
| **Final Mean ± Std** | **85.38 ± 0.35%** | **89.68 ± 0.19%**         |           |

---

## 6. How to Train & Evaluate

### 데이터셋 분할 명시 (Data Split Clarification)

- **학습(Training)**: 50,000장의 CIFAR-100 `train=True` 데이터만 모델 학습(Gradient Update)에 사용되었습니다.
- **평가(Evaluation)**: 10,000장의 공식 CIFAR-100 `train=False` 테스트 데이터는 학습에 일절 포함되지 않고 오직 최종 평가용으로만 엄격하게 분리되어 사용되었습니다.

### 6.1 학습 방법 (Training)

1. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
2. Jupyter Notebook (`run_all_wrn_28_12_2.ipynb`)을 실행합니다.
3. `run_experiment` 셀을 실행하면 학습 및 평가가 자동으로 진행됩니다.
   - 매 Epoch마다 Top-1 Acc 및 SC Acc가 출력됩니다.
   - 로그는 `log_wrn28_12_scratch.json`에 저장됩니다.
   - 최고 성능 모델은 `checkpoints/` 폴더에 자동 저장됩니다.

### 6.2 평가 방법 (Evaluation)

Jupyter Notebook 전체를 실행할 필요 없이, 학습된 최고 성능의 가중치(`.pth`)를 10,000장 테스트 셋에 대해 즉시 평가하려면 아래 스크립트를 실행하세요.

```bash
python evaluate.py
```

해당 스크립트는 10,000장의 공식 테스트 셋을 불러와 최고 성능 체크포인트를 기반으로 정확한 Top-1 및 Super-Class Accuracy를 출력합니다.

---

## 7. Reproducibility

- 사용된 Seed: **42, 77, 100**

---

## 8. File Structure

```text
├── dataloader.py                # CIFAR-100 데이터 로딩 및 전처리
├── evaluate.py                  # 10,000장 Test Set 독립 평가 스크립트
├── model(28_12).py              # 모델 구조 및 SAM 구현
├── trainer(28_12).py            # 학습 루프 및 Loss 함수
├── run_all_wrn_28_12_2.ipynb    # 전체 실행 노트북
└── README.md                    # 문서
```
