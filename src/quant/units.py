"""사전학습 모델(ResNet-18, DeiT-Tiny)용 블록 단위 양자화 토글.

6-Conv에서는 층 하나가 Conv 하나였지만, ResNet·DeiT는 블록 하나에 Conv/Linear가 여러 개 들어 있다.
그래서 "단위(unit)"를 다음처럼 정의한다(이하형 문서 11-10).
- 단위 = 블록 안의 모든 Conv/Linear 가중치 양자화기 + 블록 출력 활성값 양자화기
- 6-Conv의 "층 = W + 블록 출력 A"와 같은 규칙을 블록으로 넓힌 것

모델 코드를 다시 쓰지 않고 torchvision/timm 모델을 그대로 쓰기 위해 forward hook을 사용한다.
- 가중치: 미리 계산해 둔 양자화 가중치를 모듈의 weight에 복사해 넣는다(FP32 원본은 따로 보관)
- 활성값: 단위 출력 모듈에 hook을 걸어 출력을 fake quant한 값으로 바꾼다

잠정 가정(문서에 규정 없음, 팀 확인 필요)
- 블록 내부의 중간 활성값(예: BasicBlock의 첫 ReLU 출력, DeiT의 attention·MLP 중간값)은 양자화하지 않는다.
  LayerNorm, GELU, Softmax, Residual Add, 행렬곱을 FP32로 둔다는 문서 규칙과 맞추기 위한 선택이다.
- bias는 양자화하지 않는다(6-Conv와 동일).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .quantizer import BITS, act_qparams, fake_quant_act, fake_quant_weight, weight_qparams


@dataclass
class Unit:
    """복원·교체 실험의 한 단위."""

    name: str
    out: nn.Module  # 이 모듈의 출력이 단위의 출력 활성값
    weights: list[nn.Module]  # 가중치를 양자화할 Conv/Linear 목록
    quant_act: bool = True  # False면 출력 활성값은 양자화하지 않는다(분류기 logit 등)
    fp: list[torch.Tensor] = field(default_factory=list)  # FP32 가중치 원본


class UnitQuantizer:
    """단위별 W/A 비트를 켜고 끄는 도구. None은 FP32.

    사용 순서: UnitQuantizer(model, units) -> calibrate(batches) -> apply(w_bits, a_bits) -> model(x)
    """

    def __init__(self, model: nn.Module, units: list[Unit], bits=BITS) -> None:
        self.model, self.units, self.bits = model.eval(), units, bits
        n = len(units)
        self.a_q: list[tuple | None] = [None] * n  # (scale, zero, q_max)
        self.ranges: list[torch.Tensor] | None = None
        self.observe: list[torch.Tensor] | None = None  # 보정 중일 때 단위별 [min, max] 누적
        self.sample: list[list[torch.Tensor]] | None = None  # percentile 보정 중일 때 단위별 표본 값
        self.sample_n, self.sample_gen = 0, None
        self.capture: list[torch.Tensor] | None = None  # NMSE 계산용 단위별 출력 수집
        self.w_table: dict[int, list[list[torch.Tensor]]] = {}
        for u in units:
            u.fp = [m.weight.detach().clone() for m in u.weights]
        # 비트 수별 양자화 가중치를 미리 계산(상태 전환이 많아 매번 계산하지 않도록)
        for b in bits:
            self.w_table[b] = [[fake_quant_weight(w, *weight_qparams(w, b)) for w in u.fp] for u in units]
        self.a_table: dict[int, list[tuple | None]] = {}
        self.hooks = [u.out.register_forward_hook(self._hook(i)) for i, u in enumerate(units)]

    def _hook(self, i: int):
        def fn(_module, _inputs, output):
            if self.observe is not None:
                # 보정은 양자화를 모두 끈 FP32 순전파에서 수행한다
                lo, hi = self.observe[i]
                self.observe[i] = torch.stack([torch.minimum(lo, output.min()), torch.maximum(hi, output.max())])
            if self.sample is not None:
                # 활성값 전체는 너무 커서(stem 출력이 배치당 5천만 개) 무작위로 일부만 뽑아 CPU에 모은다
                flat = output.detach().flatten()
                idx = torch.randint(flat.numel(), (min(self.sample_n, flat.numel()),), generator=self.sample_gen)
                self.sample[i].append(flat[idx.to(flat.device)].float().cpu())
            if self.a_q[i] is not None:
                output = fake_quant_act(output, *self.a_q[i])
            if self.capture is not None:
                self.capture.append(output)  # 모든 상태에서 같은 위치(활성값 양자화기 뒤)를 비교
            return output

        return fn

    @torch.no_grad()
    def calibrate(self, batch_iter) -> list[torch.Tensor]:
        """FP32 순전파로 단위별 출력 [min, max]를 수집하고 활성값 양자화 파라미터를 만든다."""
        n = len(self.units)
        self.apply([None] * n, [None] * n)
        device = self.units[0].fp[0].device
        self.observe = [torch.tensor([float("inf"), float("-inf")], device=device) for _ in range(n)]
        for x in batch_iter:
            self.model(x)
        self.ranges, self.observe = self.observe, None
        self.set_ranges(self.ranges)
        return self.ranges

    @torch.no_grad()
    def calibrate_percentile(self, batch_iter, pct: float, n_per_batch: int = 200_000, seed: int = 0) -> list[torch.Tensor]:
        """min-max 대신 분위수로 활성값 범위를 정한다: [100 − pct, pct] 백분위수.

        min-max는 보정 이미지 몇 장의 극단값이 범위를 정해 버린다. 분위수 범위는 극단값을 잘라내는 대신
        (clipping 오차) 눈금 간격을 좁힌다(반올림 오차 감소). 기본 규칙은 여전히 min-max이고,
        이 함수는 "결론이 보정 방법에 따라 달라지는가"를 확인하는 용도다.

        근사: 배치마다 단위 출력에서 n_per_batch개를 무작위로 뽑아 모은 표본의 분위수를 쓴다.
        2,048장(8배치)이면 표본 160만 개라 99.99 백분위수 위에 약 160개가 남는다(잠정, 정밀 추정은 아님).
        """
        n = len(self.units)
        self.apply([None] * n, [None] * n)
        self.sample, self.sample_n = [[] for _ in range(n)], n_per_batch
        self.sample_gen = torch.Generator().manual_seed(seed)  # 표본 위치를 고정해 실행마다 같은 범위가 나오게 함
        for x in batch_iter:
            self.model(x)
        samples, self.sample = self.sample, None
        device = self.units[0].fp[0].device
        ranges = []
        for values in samples:
            v = torch.cat(values)
            q = torch.quantile(v, torch.tensor([1 - pct / 100, pct / 100]))
            ranges.append(q.to(device))
        self.set_ranges(ranges)
        return ranges

    def set_ranges(self, ranges: list[torch.Tensor]) -> None:
        """저장해 둔 보정 범위로 활성값 양자화 파라미터를 다시 만든다(재보정 없이 이어서 실험할 때)."""
        self.ranges = ranges
        self.a_table = {
            b: [act_qparams(r[0], r[1], b) if u.quant_act else None for r, u in zip(ranges, self.units)]
            for b in self.bits
        }

    @torch.no_grad()
    def apply(self, w_bits: list, a_bits: list) -> None:
        """단위별 비트 수 목록을 모델에 적용. None은 FP32."""
        for i, u in enumerate(self.units):
            source = u.fp if w_bits[i] is None else self.w_table[w_bits[i]][i]
            for module, w in zip(u.weights, source):
                module.weight.copy_(w)
            self.a_q[i] = None if a_bits[i] is None else self.a_table[a_bits[i]][i]

    def remove(self) -> None:
        """hook을 떼고 가중치를 FP32로 되돌린다."""
        n = len(self.units)
        self.apply([None] * n, [None] * n)
        for h in self.hooks:
            h.remove()
