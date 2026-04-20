# CIFAR-100 Classification Challenge: Team DINOFORCE TV

본 프로젝트는 CIFAR-100 데이터셋을 활용하여 세부 클래스(Fine-class) 및 슈퍼클래스(Super-class) 분류 성능을 최적화하는 모델을 개발하는 것을 목표로 합니다. 제출 가이드라인에 맞추어 모델 아키텍처, 학습 전략, 재현성 검증 방법 등을 상세히 기술합니다.

---

## 1. Project Overview

- **Dataset**: CIFAR-100 (50,000 Training / 10,000 Test images)
- **Constraint**: 'From Scratch' 학습 (사전 학습된 가중치 사용 금지), 외부 데이터 사용 금지, 단일 GPU 환경
- **Metric**: Top-1 Accuracy & Super-Class (SC) Accuracy (Top-5 예측 밀도 기준)
- **Goal**: 세부 클래스 간의 미세한 혼동(Fine-grained confusion)을 해결하고, 슈퍼클래스 구조를 안정적으로 이해하는 분류기 구축

---

## 2. Requirements (Dependencies)

프로젝트를 실행하기 위해 다음의 라이브러리 환경이 필요합니다.

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.4.0
wandb>=0.15.0 (선택: 실험 로깅용)
```

---

## 3. Model Architecture: WideResNet-28-12

- **Backbone Capacity**: 기존 WideResNet-28-10 대비 Widen Factor를 12로 확장(WRN-28-12)하여 모델의 파라미터와 표현력을 1.44배 증가시켰습니다. 이를 통해 미세한 시각적 차이를 구분하는 능력을 확보했습니다.
- **Strategic Dropout Removal**: 강력한 Data Augmentation(Mixup/CutMix)과 계층적 손실 함수가 충분한 정규화(Regularization) 역할을 수행하므로, 모델의 표현력 저하를 막기 위해 Dropout을 0.0으로 제거했습니다.

---

## 4. Key Strategies & Techniques

### 4.1 Hierarchical Loss Design

- **Hierarchical Label Smoothing**: 일반적인 One-hot 타겟 대신, 정답 클래스에 85%의 확률을 부여하고 같은 슈퍼클래스 내의 형제(Sibling) 클래스들에게 80%의 잔여 확률을 분배하는 소프트 타겟을 적용했습니다 (LS=0.15, Sibling Ratio=0.8).
- **Hybrid Objective**: "모든 오답이 똑같이 틀린 것은 아니다"라는 가설 아래, 세부 클래스를 맞히는 문제와 슈퍼클래스 내에 응집하는 문제를 동시에 해결하여 SC Accuracy를 극대화했습니다.

### 4.2 Optimization & Training

- **SAM (Sharpness-Aware Minimization)**: 손실 함수의 기울기가 평탄한(Flat) 지점을 찾아 일반화 성능을 극대화하는 최적화 기법을 적용했습니다.
- **Training Schedule**: Batch Size를 256으로 설정하고 Learning Rate를 0.2로 상향 조정한 뒤, 300 Epoch의 장기 학습과 Cosine Annealing 스케줄러를 통해 정밀한 수렴을 유도했습니다.

### 4.3 Advanced Augmentation

- **High-Intensity Mix-Aug**: 매 배치마다 Mixup 또는 CutMix가 100% 확률로 교차 적용되도록 강도를 높여 모델의 강건성(Robustness)과 과적합 억제력을 확보했습니다.

---

## 5. Final Performance

3개의 독립적인 Random Seed를 통한 300 Epoch 장기 학습 검증 결과입니다. 후보군 중 가장 강력한 성능과 일관된 슈퍼클래스 일치도를 달성했습니다.

| Model / Seed             | Top-1 Accuracy | Super-Class (SC) Accuracy | Notes        |
| :----------------------- | :------------: | :-----------------------: | :----------- |
| **WRN-28-12 (Seed 42)**  |     85.04%     |          89.46%           | Confirmed    |
| **WRN-28-12 (Seed 77)**  |     85.37%     |          89.82%           | Confirmed    |
| **WRN-28-12 (Seed 100)** |   **85.73%**   |        **89.76%**         | **Best Run** |
| **Final Mean**           |   **85.38%**   |        **89.68%**         |              |

---

## 6. How to Train & Evaluate

본 프로젝트는 Jupyter Notebook (`run_all_wrn_28_12_2.ipynb`) 환경에서 통합 실행되도록 구성되어 있습니다. 학습 스크립트 실행 시 매 Epoch마다 자동으로 Validation(평가)이 수행됩니다.

### 실행 방법

1. 구글 Colab 또는 로컬 Jupyter 환경에서 `run_all_wrn_28_12_2.ipynb`를 엽니다.
2. 상단의 필수 라이브러리 및 `.py` 모듈 로드 셀을 실행합니다. (같은 디렉토리에 `.py` 파일들이 위치해야 합니다)
3. 메인 학습 셀(`run_experiment`)을 실행하면 데이터셋 다운로드부터 모델 초기화, 학습, 평가가 순차적으로 자동 진행됩니다.
4. 평가 결과(Top-1 Acc, SC Acc)는 매 에포크마다 콘솔 및 로그 파일(`log_wrn28_12_scratch.json`)에 출력되며, 최고 성능 갱신 시 `checkpoints` 폴더에 가중치가 `.pth` 포맷으로 자동 저장됩니다.

---

## 7. Random Seed Configuration

본 프로젝트는 결과의 완벽한 재현성(Reproducibility)을 보장하기 위해 `numpy`, `torch`, `cuda`의 모든 난수 생성 알고리즘을 통제합니다. 최종 결과 도출에 사용된 공식 시드(Random Seed)는 **`42`, `77`, `100`** 총 3가지입니다.

### 시드(Seed) 변경 방법

메인 노트북 파일(`run_all_wrn_28_12_2.ipynb`) 내의 Config 딕셔너리에서 `seed` 값을 수정하여 손쉽게 변경할 수 있습니다.

```python
# [Cell] WRN-28-12 + 계층적 손실 도입 부분
_final_scratch_config = {
    'run_id': 303,
    'memo': 'SCRATCH: WRN-28-12, BS256, Drop 0.0, Hierarchical LS',
    'seed': 77, # <--- 이 부분의 숫자를 원하는 시드(예: 42, 100, 1234)로 변경합니다.

    'depth': 28,
    'widen_factor': 12,
    # ... (생략) ...
}

# 설정 변경 후 아래 함수를 실행하면 해당 시드로 완전히 새로운 학습 및 평가가 시작됩니다.
run_experiment(_final_scratch_config, device)
```

---

## 8. File Structure

```text
├── dataloader.py                # CIFAR-100 데이터셋 다운로드, 전처리 및 Superclass 매핑 로직
├── model(28_12).py              # WideResNet-28-12 모델 아키텍처 및 SAM 옵티마이저 구현체
├── trainer(28_12).py            # 학습 루프, 검증(Validation), 계층적 손실(Hierarchical Loss) 함수
├── run_all_wrn_28_12_2.ipynb    # 전체 실험 세팅, Config 관리 및 통합 실행 노트북
└── README.md                    # 프로젝트 명세서 및 실행 가이드
```
