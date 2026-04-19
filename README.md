# CIFAR-100 Classification Challenge: Team DINOFORCE

본 프로젝트는 CIFAR-100 데이터셋을 활용하여 세부 클래스(Fine-class) 및 슈퍼클래스(Super-class) 분류 성능을 최적화하는 모델을 개발하는 것을 목표로 합니다.

## 1. Project Overview

- **Dataset**: CIFAR-100 (50,000 Training / 10,000 Test images)
- **Constraint**: 'From Scratch' 학습 (사전 학습된 가중치 사용 금지), 외부 데이터 사용 금지, 단일 GPU 환경
- **Metric**: Top-1 Accuracy & Super-Class Accuracy (Top-5 밀도 기준)

## 2. Model Architecture: WideResNet-28-12

- **Structure**: 기존 WideResNet-28-10 대비 채널 수를 1.2배 확장하여 모델 용량(Capacity) 확보
- **Feature Extract**: 2-Head 구조를 채택하여 분류(Classification)와 임베딩(Embedding) 정보를 동시에 학습
- **Regularization**: Dropout 0.4 및 가중치 감쇠(Weight Decay) 적용으로 과적합 방지

## 3. Key Strategies & Techniques

### 3.1 Hierarchical Loss Design

- **Hierarchical Label Smoothing**: 세부 클래스 간의 관계를 고려하여 같은 슈퍼클래스 내 클래스들에 대해 부드러운 라벨링 적용 (LS 0.15)
- **Hybrid Loss Function**: 세부 클래스 분류를 위한 CE Loss와 슈퍼클래스 간 응집도를 높이는 SupCon(Embedding) Loss를 결합 ($Loss = \text{Hierarchical CE} + \beta \cdot L_{emb}$)

### 3.2 Optimization & Training

- **SAM Optimizer**: 손실 함수의 기울기가 평탄한(Flat) 지점을 찾아 일반화 성능을 극대화하는 Sharpness-Aware Minimization 적용
- **Training Schedule**: 300 Epoch 장기 학습 및 Cosine Annealing 스케줄러를 통한 정밀한 수렴 유도

### 3.3 Advanced Augmentation

- **High-Intensity Mix-Aug**: CutMix와 Mixup을 100% 확률로 적용하여 모델의 강건성(Robustness) 확보
- **Feature Preserving**: ColorJitter 등을 통해 물체의 형태는 보존하면서 질감과 색상 변화에 대한 대응력 강화

## 4. Final Performance

| Metric                   | Result    |
| :----------------------- | :-------- |
| **Top-1 Accuracy**       | **85.3%** |
| **Super-Class Accuracy** | **89.1%** |

## 5. File Structure

- `dataloader.py`: CIFAR-100 데이터 로딩 및 계층적 맵 추출
- `model(28_12).py`: WideResNet-28-12 아키텍처 및 SAM 구현
- `trainer(28_12).py`: 학습 루프 및 계층적 손실 계산 로직
- `run_all_wrn_28_12_2.ipynb`: 전체 실험 실행 및 결과 시각화 노트북
