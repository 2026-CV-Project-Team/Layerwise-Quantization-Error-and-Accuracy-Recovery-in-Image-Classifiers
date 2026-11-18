"""오차 전달 분석: 한 단위에서 생긴 양자화 오차가 최종 출력까지 얼마나 전달되는지 잰다.

배경 (기존 결과를 다시 보고 세운 가설, 11-18)
- 단위 하나만 양자화했을 때 그 단위 출력의 오차(nmse_only)는 단위마다 비슷했다
  (ResNet-18 W4A4 0.09~0.21, DeiT-Tiny W5A5 0.07~0.13).
- 그런데 같은 상태에서 최종 출력(logit) 오차는 단위마다 10배 넘게 달랐다(ResNet-18 stem 0.36, layer4.1 0.02).
- 그래서 "오차가 얼마나 큰가"보다 "그 오차가 출력까지 얼마나 살아서 가는가"가 복원 효과를 정한다고 보고,
  이를 직접 잰다. 누적 NMSE 순위와 복원 이득 순위가 왜 어긋나는지 설명하려는 실험이다.

재는 것 (모두 rank 앞쪽 2,000장, 순위·분석용이라 test는 쓰지 않음)
1. 전달 행렬: 단위 i만 양자화한 상태에서 모든 단위 j의 출력 오차 NMSE[i, j]. 마지막 열(분류기 출력)이 최종 출력 오차다.
   - 증폭률 = NMSE[i, 마지막] / NMSE[i, i]  (i에서 생긴 상대 오차가 출력에서 몇 배가 되는가)
   - 전부 양자화한 상태의 행도 같이 재서, 누적 오차 중 그 단위에서 새로 생긴 몫과 넘어온 몫을 비교한다
   - DeiT는 블록 출력을 cls 토큰과 patch 토큰으로 나눠서도 잰다(분류기는 cls 토큰만 쓰기 때문)
2. 가산성: 단위별 최종 출력 오차를 그냥 더한 값이 여러 단위를 함께 양자화했을 때의 실제 오차와 맞는가
   - 단위마다 비트를 다르게 준 무작위 조합을 평가해서, 단일 측정값의 합으로 만든 예측과 비교한다
   - 맞는 구간에서는 2^N 조합을 다 돌리지 않고 단일 측정 N×4번으로 조합의 오차를 예측할 수 있다
3. W/A 분리: 단위 i의 W만, A만 양자화했을 때의 최종 출력 오차. 그 단위의 오차가 어느 쪽에서 오는지 본다
4. 보정 방법: 활성값 범위를 min-max 대신 분위수(99.99, 99.9 백분위수)로 잡아도 같은 양상인지 본다
   - min-max는 보정 이미지의 극단값에 좌우된다(보정 세트 반복 측정에서 A 4비트 정확도가 3~4%p 흔들림)
   - 결론이 min-max라는 선택 때문에 생긴 것인지 확인하는 용도이고, 기본 규칙을 바꾸는 것은 아니다

잠정 가정 (팀 확인 필요)
- 평가 이미지는 rank 앞쪽 2,000장. 노트북 발열 때문에 전체 5,000장 대신 줄였다(정확도 표준오차 약 1%p).
  NMSE·KL은 이미지별 평균이라 2,000장으로도 충분히 안정적이지만, 정확도 차이 1~2%p는 해석하지 않는다.
- 전달 행렬과 복원은 ResNet-18 W4A4·W5A5, DeiT-Tiny W5A5에서 잰다(DeiT W4A4는 정확도 0.06%로 무너져서 제외).
- 무작위 조합은 비트 묶음 4종({6,8}, {5,6,8}, {4,5,6,8}, {4,8})에서 ResNet-18 15개씩, DeiT-Tiny 10개씩.
  W와 A는 같은 비트(기존 Mixed와 동일). {4,8}은 4비트 단위 수를 0~N에서 고르게 뽑는다.
- 분위수 보정은 표본 근사다(units.calibrate_percentile 참고).

출력: results/imagenet_propagation/{모델}/ 아래 matrix.csv, evals.csv, units.csv, ranges.json
집계: python -m src.propagation_summary

실행 예 (M4 MPS 기준 ResNet-18 15분, DeiT-Tiny 45분 정도)
    python -m src.exp_imagenet_propagation --models resnet18 deit_tiny
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .exp_imagenet import ROOT, pick_device
from .exp_imagenet_output import output_error
from .imagenet_data import load_split, to_input
from .imagenet_models import MODELS, load_model, unit_param_counts
from .quant.quantizer import BITS
from .quant.units import UnitQuantizer

N_SUB = 2000  # rank 앞쪽에서 쓰는 장수
MATRIX_BITS = {"resnet18": [4, 5], "deit_tiny": [5]}  # 전달 행렬·복원을 재는 비트 (min-max)
PCT_BITS = {"resnet18": [4], "deit_tiny": [5]}  # 분위수 보정에서 다시 재는 비트
SPLIT_BIT = {"resnet18": 4, "deit_tiny": 5}  # W/A 분리를 재는 비트
CALIBS = ("minmax", "p99.99", "p99.9")  # p99.9는 전 단위 동일 비트 조건만 잰다
FAMILIES = {"b68": (6, 8), "b568": (5, 6, 8), "b4568": (4, 5, 6, 8), "b48": (4, 8)}
N_COMBO = {"resnet18": 15, "deit_tiny": 10}  # 비트 묶음마다 뽑는 무작위 조합 수


def make_combos(n: int, per_family: int, seed: int = 0) -> list[tuple[str, list[int]]]:
    """단위별 비트 무작위 조합. seed를 고정해서 다시 실행해도 같은 조합이 나온다."""
    rng = np.random.default_rng(seed)
    out = []
    for family, bits in FAMILIES.items():
        for _ in range(per_family):
            if family == "b48":
                # 4비트 단위 수를 0~N에서 고르게 뽑는다. 단위마다 반반으로 뽑으면 4비트가 절반쯤인 조합만 나와서
                # "거의 다 8비트"처럼 Pareto에서 중요한 구간이 비기 때문이다
                v = np.full(n, 8)
                v[rng.choice(n, int(rng.integers(0, n + 1)), replace=False)] = 4
            else:
                v = rng.choice(bits, n)
            out.append((family, [int(b) for b in v]))
    return out


def text(bits: list) -> str:
    """비트 목록을 CSV에 넣을 문자열로. None(FP32)은 32로 적는다."""
    return " ".join(str(32 if b is None else b) for b in bits)


class Runner:
    """모델 하나의 측정을 묶어 둔 것. FP32 logit을 한 번 구해 두고 모든 상태를 그것과 비교한다."""

    def __init__(self, name: str, device, n_images: int = N_SUB) -> None:
        self.name, self.device = name, device
        self.model, self.units = load_model(name)
        self.model.to(device)
        self.q = UnitQuantizer(self.model, self.units)
        self.n = len(self.units)
        self.off = [None] * self.n
        x, y = load_split(name, "rank")
        self.x, self.y = x[:n_images], y[:n_images]
        self.calib_x, _ = load_split(name, "calib")
        self.ref: torch.Tensor | None = None
        self.calib = ""
        self.evals: list[dict] = []
        self.matrix: list[dict] = []

    def calib_batches(self):
        return (to_input(self.calib_x[i : i + 256], self.device) for i in range(0, len(self.calib_x), 256))

    @torch.no_grad()
    def logits(self, w: list, a: list) -> torch.Tensor:
        self.q.apply(w, a)
        out = [self.model(to_input(self.x[i : i + 250], self.device)).float().cpu() for i in range(0, len(self.x), 250)]
        return torch.cat(out)

    def record(self, kind: str, w: list, a: list, out: torch.Tensor, **extra) -> dict:
        """한 상태의 정확도와 최종 출력 오차(FP32 logit 대비)를 기록."""
        nmse, kl = output_error(self.ref, out)
        row = {"model": self.name, "calib": self.calib, "kind": kind, "w": text(w), "a": text(a),
               "acc": float((out.argmax(1) == self.y).float().mean() * 100), "out_nmse": nmse, "out_kl": kl, **extra}
        self.evals.append(row)
        return row

    def evaluate(self, kind: str, w: list, a: list, **extra) -> dict:
        return self.record(kind, w, a, self.logits(w, a), **extra)

    @torch.no_grad()
    def propagate(self, bit: int) -> None:
        """전달 행렬. 상태 = [전부 양자화] + [단위 i만 양자화] N개. 배치마다 FP32 출력을 기준으로 바로 비교한다.

        capture는 각 단위의 활성값 양자화기 뒤 값이라, 단위 i만 양자화한 상태에서 j ≠ i의 오차는
        순수하게 앞에서 넘어온 것이다. 마지막 단위(분류기)의 출력이 logit이므로 같은 순전파에서
        최종 출력 오차와 정확도도 같이 얻는다.
        """
        n, q = self.n, self.q
        states = [([bit] * n, [bit] * n)]
        for i in range(n):
            w, a = list(self.off), list(self.off)
            w[i], a[i] = bit, bit
            states.append((w, a))
        total = torch.zeros(len(states), n)
        token = torch.full((len(states), n, 2), float("nan"))  # DeiT 블록 출력의 (cls, patch) 오차
        outs: list[list[torch.Tensor]] = [[] for _ in states]

        def nmse(r: torch.Tensor, o: torch.Tensor) -> float:
            r, o = r.flatten(1), o.flatten(1)
            return float(((r - o).pow(2).sum(1) / r.pow(2).sum(1).clamp(min=1e-12)).sum())

        for i in range(0, len(self.x), 100):
            batch = to_input(self.x[i : i + 100], self.device)
            q.apply(self.off, self.off)
            q.capture = []
            self.model(batch)
            ref = q.capture
            for s, (w, a) in enumerate(states):
                q.apply(w, a)
                q.capture = []
                self.model(batch)
                for k, (r, o) in enumerate(zip(ref, q.capture)):
                    total[s, k] += nmse(r, o)
                    if r.ndim == 3 and r.shape[1] == 197:  # DeiT 블록 출력 (배치, 토큰 197, 차원). 0번이 cls
                        prev = torch.nan_to_num(token[s, k])
                        token[s, k] = prev + torch.tensor([nmse(r[:, :1], o[:, :1]), nmse(r[:, 1:], o[:, 1:])])
                outs[s].append(q.capture[-1].float().cpu())
        q.capture = None
        q.apply(self.off, self.off)
        total, token = total / len(self.x), token / len(self.x)
        for s, (w, a) in enumerate(states):
            src = s - 1  # -1 = 전부 양자화한 상태
            out = torch.cat(outs[s])
            if src < 0:
                self.record("uniform_m", w, a, out, bit=bit)  # logits()로 잰 uniform과 같은 값인지 확인용
            else:
                self.record("single_wa", w, a, out, bit=bit, unit_idx=src, unit=self.units[src].name)
            for k, u in enumerate(self.units):
                self.matrix.append({"model": self.name, "calib": self.calib, "bit": bit, "src_idx": src,
                                    "src": "all" if src < 0 else self.units[src].name, "unit_idx": k, "unit": u.name,
                                    "nmse": float(total[s, k]), "nmse_cls": float(token[s, k, 0]),
                                    "nmse_patch": float(token[s, k, 1])})

    def run(self, out_dir: Path, rest: float) -> None:
        n, units = self.n, self.units
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"model": self.name, "unit_idx": range(n), "unit": [u.name for u in units],
                      "params": unit_param_counts(units)}).to_csv(out_dir / "units.csv", index=False)
        saved_ranges = {}

        def pause(label: str) -> None:
            print(self.name, self.calib, label, f"{time.time() - t0:.0f}s", flush=True)
            self.save(out_dir)  # 단계가 끝날 때마다 저장해서 중간에 멈춰도 그때까지 결과가 남게 한다
            if rest:
                time.sleep(rest)  # 발열 관리용 쉬는 시간

        t0 = time.time()
        for calib in CALIBS:
            self.calib = calib
            ranges = (self.q.calibrate(self.calib_batches()) if calib == "minmax"
                      else self.q.calibrate_percentile(self.calib_batches(), float(calib[1:])))
            saved_ranges[calib] = {u.name: [float(r[0]), float(r[1])] for u, r in zip(units, ranges)}
            (out_dir / "ranges.json").write_text(json.dumps(saved_ranges, indent=1))
            if self.ref is None:
                self.ref = self.logits(self.off, self.off)  # FP32 기준. 보정 방법과 무관하다
                self.record("fp32", self.off, self.off, self.ref)
            for b in BITS:  # 전 단위 동일 비트
                self.evaluate("uniform", [b] * n, [b] * n, bit=b)
            pause("uniform")
            if calib == "p99.9":
                continue
            for b in (MATRIX_BITS if calib == "minmax" else PCT_BITS)[self.name]:
                self.propagate(b)
                pause(f"matrix W{b}A{b}")
                for i, u in enumerate(units):  # 전부 양자화한 상태에서 단위 i만 FP32로 복원
                    w, a = [b] * n, [b] * n
                    w[i], a[i] = None, None
                    self.evaluate("restore", w, a, bit=b, unit_idx=i, unit=u.name)
                pause(f"restore W{b}A{b}")
            if calib != "minmax":
                continue
            # 가산성 예측에 필요한 나머지 비트의 단일 측정 (전달 행렬에서 이미 잰 비트는 건너뜀)
            for b in BITS:
                if b in MATRIX_BITS[self.name]:
                    continue
                for i, u in enumerate(units):
                    w, a = list(self.off), list(self.off)
                    w[i], a[i] = b, b
                    self.evaluate("single_wa", w, a, bit=b, unit_idx=i, unit=u.name)
                pause(f"single W{b}A{b}")
            # W만, A만
            b = SPLIT_BIT[self.name]
            for kind in ("single_w", "single_a"):
                for i, u in enumerate(units):
                    w, a = list(self.off), list(self.off)
                    if kind == "single_w":
                        w[i] = b
                    else:
                        a[i] = b
                    self.evaluate(kind, w, a, bit=b, unit_idx=i, unit=u.name)
            pause(f"split {b}")
            # 무작위 조합
            for family, bits in make_combos(n, N_COMBO[self.name]):
                self.evaluate("combo", bits, bits, family=family)
            pause("combo")
        self.q.remove()

    def save(self, out_dir: Path) -> None:
        pd.DataFrame(self.evals).to_csv(out_dir / "evals.csv", index=False)
        if self.matrix:
            pd.DataFrame(self.matrix).to_csv(out_dir / "matrix.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--out", default="results/imagenet_propagation")
    p.add_argument("--rest", type=float, default=0, help="단계 사이에 쉬는 시간(초). 노트북 발열 관리용")
    p.add_argument("--n", type=int, default=N_SUB, help="rank 앞쪽에서 쓰는 장수 (동작 확인용으로 줄일 때)")
    args = p.parse_args()
    device = pick_device()
    for name in args.models:
        Runner(name, device, args.n).run(ROOT / args.out / name, args.rest)


if __name__ == "__main__":
    main()
