# W6A6 양자화 방식별 층간 오차 전달과 복원

기존 [오차 전달 분석(PR #23)](https://github.com/2026-CV-Project-Team/Layerwise-Quantization-Error-and-Accuracy-Recovery-in-Image-Classifiers/pull/23)의 측정 방법을 W6A6의 세 가지 양자화 규칙에 적용했다. 같은 비트 수에서도 오차가 크게 전달되는 단위와 복원으로 이득을 보는 단위가 모델과 규칙에 따라 달라지는지 비교했다.

## 실험 설정

- ImageNet 사전학습 ResNet-18과 DeiT-Tiny를 사용했다. 보정은 seed 0의 2,048장과 min-max 범위로 고정했다. 가중치는 출력 채널별 대칭 양자화, 활성값은 블록 출력의 텐서별 비대칭 양자화를 사용했다. 분류기의 가중치는 양자화 대상이며 logit은 FP32로 유지했다.
- `round`는 가장 가까운 정수로 반올림하고, `trunc_zero`는 0 방향으로 절삭하며, `floor`는 음의 무한대 방향으로 내린다. 이 규칙은 가중치와 활성값에 적용했고 스케일과 제로포인트 계산은 공유했다.
- rank 앞 2,000장에서 한 단위의 W/A만 양자화해 후속 단위와 최종 logit의 NMSE를 쟀다. 한 단위의 W만 또는 A만 양자화한 결과도 기록했다. 별도 복원 평가에서는 전체를 W6A6로 양자화한 상태에서 한 단위만 FP32로 되돌렸다.
- 각 모델·방식에서 rank 정확도가 가장 높았던 단위 하나를 고정해 test 10,000장에서 평가했다. ResNet-18 round의 stem과 layer1.0은 rank 정확도 69.00%로 같았고, 그중 최종 logit NMSE가 작은 stem을 선택했다.

## Test 결과

| 모델 | 방식 | 복원 단위 | FP32 | 전체 양자화 | 단위 복원 | 변화 |
|---|---|---|---:|---:|---:|---:|
| ResNet-18 | round | stem | 69.19% | 67.31% | 68.52% | +1.21%p |
| ResNet-18 | trunc_zero | layer3.0 | 69.19% | 38.06% | 46.59% | +8.53%p |
| ResNet-18 | floor | layer4.0 | 69.19% | 0.09% | 0.16% | +0.07%p |
| DeiT-Tiny | round | block9 | 71.90% | 62.17% | 63.64% | +1.47%p |
| DeiT-Tiny | trunc_zero | block7 | 71.90% | 0.11% | 0.09% | −0.02%p |
| DeiT-Tiny | floor | patch_embed | 71.90% | 39.20% | 62.24% | +23.04%p |

## Rank에서 관찰한 오차 전달

| 모델 | 방식 | 최종 logit NMSE가 가장 큰 단독 양자화 단위 | 최대 NMSE | 최종 NMSE와 복원 이득의 순위 상관 |
|---|---|---|---:|---:|
| ResNet-18 | round | stem | 0.0277 | 0.488 |
| ResNet-18 | trunc_zero | stem | 0.2834 | 0.394 |
| ResNet-18 | floor | stem | 1.2815 | 0.013 |
| DeiT-Tiny | round | block11 | 0.1353 | 0.854 |
| DeiT-Tiny | trunc_zero | block11 | 0.4016 | 0.042 |
| DeiT-Tiny | floor | patch_embed | 0.7881 | 0.512 |

- ResNet-18은 세 방식 모두 stem의 단독 양자화가 최종 logit을 가장 크게 바꿨다. 그러나 trunc_zero 전체 양자화에서는 layer3.0을 복원했을 때 정확도가 가장 많이 올랐다. 한 단위에서 시작한 오차와 여러 단위가 양자화된 상태의 복원 효과는 각각의 조건에서 해석해야 한다.
- DeiT-Tiny에서는 가장 큰 최종 오차를 만든 단위가 round 및 trunc_zero의 block11에서 floor의 patch_embed로 바뀌었다. floor로 patch_embed의 W만 양자화해도 최종 logit NMSE가 0.7422였고, 이 단위 전체를 FP32로 복원한 test 정확도는 23.04%p 올랐다.
- ResNet-18의 단위별 A-only 결과는 trunc_zero와 floor에서 같았다. 활성값을 양자화하는 지점의 ReLU 이후 출력이 비음수라는 점과 부합한다. DeiT-Tiny에서는 단위별 A-only 최종 NMSE 평균이 trunc_zero 0.1587, floor 0.0484로 달랐다.

## 재현과 결과 파일

[README의 ImageNet 준비 절](../README.md#imagenet-사전학습-모델-resnet-18-deit-tiny)에 따라 seed 0의 보정, rank, test 캐시를 준비한 뒤 저장소 루트에서 실행한다.

```bash
python -m src.exp_imagenet_propagation_rounding --models resnet18 deit_tiny --modes round trunc_zero floor --n 2000
python -m src.exp_imagenet_propagation_rounding_test
```

- [rank 조건별 요약](../results/imagenet_propagation_rounding/summary/rank_summary.csv)과 [rank 단위별 지표](../results/imagenet_propagation_rounding/summary/rank_units.csv)를 함께 제공한다. rank 스크립트의 모델·방식별 `matrix.csv`에는 중간 단위와 DeiT의 CLS/patch 토큰 NMSE도 기록된다.
- [test 조건별 요약](../results/imagenet_propagation_rounding/summary/test_summary.csv)은 rank에서 확정한 후보의 전체 양자화 대비 변화를 담는다. test 스크립트는 모델·방식별 결과를 `results/imagenet_propagation_rounding_test/`에 기록한다.

## 해석 범위

보정 seed 하나, 비트 수 하나, CNN과 ViT 각각 한 모델에서 수행한 모의 양자화 결과다. 단위별 순위 상관은 탐색 지표이며 정확도 차이의 신뢰구간은 산출하지 않았다. test에서 본 정확도 변화는 이 설정과 복원 후보에 해당한다.
