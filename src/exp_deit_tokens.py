"""DeiT-Tiny 블록 출력의 활성값 범위를 토큰·채널별로 나눴을 때 정확도가 얼마나 돌아오는지 본다.

배경 (오차 전달 분석 결과, README 참고)
- 블록 출력(토큰 197개 x 192차원) 전체에 범위를 하나만 쓰면 cls 토큰과 patch 토큰을 둘 다 맞추지 못하는 것으로 보였다.
  min-max 범위에서는 뒤쪽 블록(block9~11)의 cls 토큰 오차가 patch의 3~4배였고,
  99.99 백분위수 범위에서는 앞쪽 블록(block0~5)의 cls 토큰 오차가 0.42~0.49까지 커졌다.
- 분류기는 cls 토큰만 읽는다. 그래서 "cls 토큰에만 범위를 따로 주면 되는가"를 확인한다.
- 원인이 토큰이 아니라 채널 간 범위 차이일 수도 있어서(FQ-ViT가 지적한 문제) 채널별 범위도 같이 비교한다.

비교하는 범위 방식 (W는 그대로 채널별 대칭 min-max, 바꾸는 건 블록 출력 A의 범위뿐)
- tensor_minmax:  블록 출력 전체에 범위 하나 (지금까지의 기본 규칙)
- tensor_p99.99:  범위 하나, 99.99 백분위수
- cls_minmax:     cls 토큰 범위 1개 + patch 토큰 범위 1개, 둘 다 min-max  (블록당 scale 1개 추가)
- cls_mixed:      cls 토큰은 min-max, patch 토큰은 99.99 백분위수          (블록당 scale 1개 추가)
- token_minmax:   토큰 위치마다 범위 1개 (197개). 기존 연구의 토큰별 양자화에 해당하는 상한선 역할
- channel_minmax: 채널마다 범위 1개 (192개)
- clsch_minmax:   cls 토큰 범위 1개 + patch 토큰은 채널마다 1개 (193개). 위 두 방식을 합친 것
- clsch2_minmax:  cls 토큰도 채널마다 1개 + patch 토큰도 채널마다 1개 (384개)
  (뒤 두 개는 rank 결과를 보고 11-18에 추가. cls 분리와 채널 분리가 서로 다른 원인을 잡는지 보려는 것)

잠정 가정 (팀 확인 필요)
- patch_embed 출력에는 cls 토큰이 없어서 cls_* 방식에서는 범위 하나를 그대로 쓴다.
- 토큰별·채널별 범위를 실제 정수 연산에서 쓰려면 뒤 연산(LayerNorm, 행렬곱)에서 scale을 맞춰 줘야 한다.
  여기서는 모의 양자화라 그 비용은 따지지 않고 정확도만 본다.
- 방식 비교는 rank 앞쪽 2,000장으로 하고, test는 --test로 고른 방식만 따로 평가한다(test를 보고 방식을 고르지 않음).
- cls_mixed의 patch 분위수는 배치마다 20만 개를 뽑은 표본으로 구한 근사다.

출력: results/deit_tokens/summary/
- evals.csv:  방식·비트별 정확도와 최종 출력 오차 (set = rank2000 또는 test)
- ranges.csv: 블록별 cls 토큰과 patch 토큰의 min-max 범위, patch의 99.99 백분위수 (cls 토큰 값이 실제로 더 큰지 확인용)
- tokens.csv: 전부 W5A5로 양자화한 상태에서 블록별 cls / patch 토큰 오차 (방식별)

실행 예
    python -m src.exp_deit_tokens                                   # rank 앞쪽 2,000장, 7분 정도
    python -m src.exp_deit_tokens --test --variants tensor_minmax cls_minmax   # 고른 방식만 test 1만 장
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from .exp_imagenet import ROOT, pick_device
from .exp_imagenet_output import output_error
from .imagenet_data import load_split, to_input
from .imagenet_models import load_model
from .quant.quantizer import BITS
from .quant.units import UnitQuantizer

MODEL = "deit_tiny"
VARIANTS = ("tensor_minmax", "tensor_p99.99", "cls_minmax", "cls_mixed", "token_minmax", "channel_minmax",
            "clsch_minmax", "clsch2_minmax")
PCT = 99.99
N_SUB = 2000


def has_cls(x: torch.Tensor) -> bool:
    """블록 출력 (배치, 토큰 197, 차원)인가. patch_embed 출력은 토큰이 196개라 cls가 없다."""
    return x.ndim == 3 and x.shape[1] == 197


@torch.no_grad()
def collect(model, units, batches, seed: int = 0) -> list[dict]:
    """FP32 순전파에서 단위 출력의 통계를 모은다. 단위마다 dict 하나.

    - token:   토큰 위치별 (min, max), shape (2, 1, T, 1)
    - channel: 채널별 (min, max), shape (2, 1, 1, C)
    - channel_cls, channel_patch: 같은 것을 cls 토큰만, patch 토큰만으로 구한 것 (cls가 있는 블록 출력만)
    - patch_q: patch 토큰 값의 분위수 [100 − PCT, PCT] (cls가 없으면 전체 값)
    방식별 범위는 이 통계에서 ranges()가 만든다(순전파를 방식마다 다시 하지 않으려고).
    """
    stats = [{"token": None, "channel": None, "channel_cls": None, "channel_patch": None, "sample": []} for _ in units]
    gen = torch.Generator().manual_seed(seed)  # 분위수 표본 위치 고정

    def hook(i):
        def fn(_module, _inputs, out):
            if out.ndim != 3:
                return
            s = stats[i]
            token = torch.stack([out.amin(dim=(0, 2), keepdim=True), out.amax(dim=(0, 2), keepdim=True)])
            channel = torch.stack([out.amin(dim=(0, 1), keepdim=True), out.amax(dim=(0, 1), keepdim=True)])
            parts = [("token", token), ("channel", channel)]
            if has_cls(out):
                for key, part in (("channel_cls", out[:, :1]), ("channel_patch", out[:, 1:])):
                    parts.append((key, torch.stack([part.amin(dim=(0, 1), keepdim=True), part.amax(dim=(0, 1), keepdim=True)])))
            for key, new in parts:
                old = s[key]
                s[key] = new if old is None else torch.stack([torch.minimum(old[0], new[0]), torch.maximum(old[1], new[1])])
            flat = (out[:, 1:] if has_cls(out) else out).flatten()
            idx = torch.randint(flat.numel(), (200_000,), generator=gen)
            s["sample"].append(flat[idx.to(flat.device)].float().cpu())

        return fn

    hooks = [u.out.register_forward_hook(hook(i)) for i, u in enumerate(units)]
    for x in batches:
        model(x)
    for h in hooks:
        h.remove()
    for s in stats:
        if s["sample"]:
            q = torch.quantile(torch.cat(s.pop("sample")), torch.tensor([1 - PCT / 100, PCT / 100]))
            s["patch_q"] = q.to(s["token"].device)
    return stats


def ranges(variant: str, stats: list[dict], tensor_ranges: dict[str, list[torch.Tensor]]) -> list:
    """방식별 단위 범위. 원소는 (min, max)이고 단위 출력 (배치, 토큰, 차원)에 broadcast되는 shape이다."""
    kind, rule = variant.split("_")
    if kind == "tensor":
        return tensor_ranges[rule]
    out = []
    for s, base in zip(stats, tensor_ranges["minmax"]):
        if s["token"] is None:  # 분류기 출력(logit)은 양자화하지 않으므로 값만 채워 둔다
            out.append(base)
            continue
        token = s["token"]  # (2, 1, T, 1)
        if kind == "token":
            out.append(token)
        elif kind == "channel":
            out.append(s["channel"])
        elif token.shape[2] != 197:  # cls 토큰이 없는 patch_embed: cls_*는 범위 하나, clsch*는 채널별
            out.append(s["channel"] if kind.startswith("clsch") else base)
        elif kind.startswith("clsch"):
            # (2, 1, 197, C): patch 토큰 196개는 채널별 범위를 같이 쓰고, 0번(cls)만 따로 둔다
            r = s["channel_patch"].expand(2, 1, 197, -1).clone()
            r[:, :, 0, :] = s["channel_cls"][:, :, 0, :] if kind == "clsch2" else token[:, :, 0, :]
            out.append(r)
        else:
            r = token.clone()
            # 0번(cls)은 자기 min-max를 그대로 두고, patch 196개는 한 범위로 묶는다
            lo, hi = (token[0, :, 1:].min(), token[1, :, 1:].max()) if rule == "minmax" else s["patch_q"]
            r[0, :, 1:], r[1, :, 1:] = lo, hi
            out.append(r)
    return out


@torch.no_grad()
def logits(model, x, device) -> torch.Tensor:
    return torch.cat([model(to_input(x[i : i + 250], device)).float().cpu() for i in range(0, len(x), 250)])


@torch.no_grad()
def token_error(model, q: UnitQuantizer, bit: int, x, device) -> list[tuple[float, float]]:
    """전부 양자화한 상태에서 블록별 (cls 토큰 NMSE, patch 토큰 NMSE). 이미지별로 계산해 평균한다."""
    n = len(q.units)
    total = torch.zeros(n, 2)

    def nmse(r, o):
        r, o = r.flatten(1), o.flatten(1)
        return float(((r - o).pow(2).sum(1) / r.pow(2).sum(1).clamp(min=1e-12)).sum())

    for i in range(0, len(x), 100):
        batch = to_input(x[i : i + 100], device)
        q.apply([None] * n, [None] * n)
        q.capture = []
        model(batch)
        ref = q.capture
        q.apply([bit] * n, [bit] * n)
        q.capture = []
        model(batch)
        for k, (r, o) in enumerate(zip(ref, q.capture)):
            if has_cls(r):
                total[k] += torch.tensor([nmse(r[:, :1], o[:, :1]), nmse(r[:, 1:], o[:, 1:])])
    q.capture = None
    return [(float(a), float(b)) for a, b in total / len(x)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--test", action="store_true", help="rank 2,000장 대신 test 1만 장으로 평가 (고른 방식의 최종 확인용)")
    p.add_argument("--n", type=int, default=N_SUB, help="rank 앞쪽에서 쓰는 장수 (동작 확인용으로 줄일 때)")
    p.add_argument("--out", default="results/deit_tokens")
    args = p.parse_args()
    device = pick_device()
    model, units = load_model(MODEL)
    model.to(device)
    q = UnitQuantizer(model, units)
    n = len(units)
    off = [None] * n
    calib_x, _ = load_split(MODEL, "calib")
    batches = lambda: (to_input(calib_x[i : i + 256], device) for i in range(0, len(calib_x), 256))  # noqa: E731
    tensor_ranges = {"p99.99": q.calibrate_percentile(batches(), PCT), "minmax": q.calibrate(batches())}
    stats = collect(model, units, batches())

    if args.test:
        x, y = load_split(MODEL, "test")
        set_name = "test"
    else:
        x, y = load_split(MODEL, "rank")
        x, y = x[: args.n], y[: args.n]
        set_name = f"rank{args.n}"
    summary = ROOT / args.out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    q.apply(off, off)
    ref = logits(model, x, device)
    acc_fp32 = float((ref.argmax(1) == y).float().mean() * 100)
    print(set_name, "FP32", round(acc_fp32, 2), flush=True)

    rows, token_rows = [], []
    t0 = time.time()
    for variant in args.variants:
        q.set_ranges(ranges(variant, stats, tensor_ranges))
        for b in BITS:
            q.apply([b] * n, [b] * n)
            out = logits(model, x, device)
            nmse, kl = output_error(ref, out)
            rows.append({"set": set_name, "variant": variant, "bit": b, "acc_fp32": acc_fp32,
                         "acc": float((out.argmax(1) == y).float().mean() * 100), "out_nmse": nmse, "out_kl": kl})
            print(variant, f"W{b}A{b}", round(rows[-1]["acc"], 2), f"{time.time() - t0:.0f}s", flush=True)
        if not args.test:  # 토큰별 오차는 원인 확인용이라 rank에서만 잰다
            for u, (e_cls, e_patch) in zip(units, token_error(model, q, 5, x, device)):
                if u.name.startswith("block"):
                    token_rows.append({"variant": variant, "bit": 5, "unit": u.name, "nmse_cls": e_cls, "nmse_patch": e_patch})
    q.remove()

    new = pd.DataFrame(rows)
    path = summary / "evals.csv"
    if path.exists():  # rank와 test, 방식을 나눠 실행해도 이전 행이 남도록 합친다
        old = pd.read_csv(path)
        keys = set(zip(new.set, new.variant))
        old = old[[(s, v) not in keys for s, v in zip(old.set, old.variant)]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    if token_rows:
        tokens = pd.DataFrame(token_rows)
        if (summary / "tokens.csv").exists():  # 방식을 나눠 실행해도 이전 행이 남도록 합친다
            old = pd.read_csv(summary / "tokens.csv")
            tokens = pd.concat([old[~old.variant.isin(tokens.variant.unique())], tokens], ignore_index=True)
        tokens.to_csv(summary / "tokens.csv", index=False)
    # 블록별 cls / patch 범위: cls 토큰 값이 patch보다 실제로 큰지 직접 확인하는 표
    range_rows = []
    for u, s in zip(units, stats):
        if s["token"] is not None and s["token"].shape[2] == 197:
            t = s["token"]
            range_rows.append({"unit": u.name, "cls_min": float(t[0, :, 0].min()), "cls_max": float(t[1, :, 0].max()),
                               "patch_min": float(t[0, :, 1:].min()), "patch_max": float(t[1, :, 1:].max()),
                               "patch_p_low": float(s["patch_q"][0]), "patch_p_high": float(s["patch_q"][1])})
    pd.DataFrame(range_rows).to_csv(summary / "ranges.csv", index=False)
    print(new[new.set == set_name].pivot(index="variant", columns="bit", values="acc").reindex(args.variants).round(2))


if __name__ == "__main__":
    main()
