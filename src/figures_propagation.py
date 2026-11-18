"""오차 전달 분석 그림. propagation_summary가 만든 집계 CSV만 읽는다(모델을 다시 돌리지 않음).

    python -m src.propagation_summary
    python -m src.figures_propagation     # -> results/imagenet_propagation/figures/*.png

그림 목록
- fig6_propagation: 전달 행렬. 단위 하나만 양자화했을 때 뒤쪽 단위들과 최종 출력에서 잰 오차 (heatmap)
- fig7_mechanism:   모델별로 단위 출력 오차 / 그 오차가 만든 최종 출력 오차 / 복원 이득을 나란히 그린 것
- fig8_additivity:  무작위 비트 조합의 최종 출력 오차. 단일 측정의 합으로 한 예측 대 실제
- fig9_tokens:      DeiT 블록 하나만 양자화했을 때 그 블록 출력의 오차를 cls 토큰과 patch 토큰으로 나눈 것 (보정 방법별)

그리는 원칙은 figures.py와 같다(축 하나, 글자는 영어, 색은 같은 조합).
- fig6은 크기를 보는 그림이라 한 색의 밝기 단계만 쓴다. 뒤 단위의 오차가 앞 단위로 가는 일은 없어서 왼쪽 아래는 비워 둔다.
- fig7은 행마다 단위가 달라서(NMSE, %p) 칸을 나누고, 모델끼리는 y축을 맞추지 않는다(모델 간 크기 비교가 목적이 아님).
- fig8은 값 범위가 100배 넘게 벌어져서 두 축 모두 log로 그린다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import BLUE, GRID, INK, MUTED, ORANGE, SEQ  # 가져오면서 figures.py의 글꼴·축 설정도 같이 적용된다

MODEL_NAMES = {"resnet18": "ResNet-18", "deit_tiny": "DeiT-Tiny"}
MAIN = {"resnet18": 4, "deit_tiny": 5}  # 본문 그림에 쓰는 비트 (DeiT W4A4는 정확도가 무너져서 W5A5)


def short(unit: str) -> str:
    """축 눈금용 짧은 이름."""
    return unit.replace("layer", "L").replace("block", "B").replace("patch_embed", "patch")


def fig_propagation(matrix: pd.DataFrame, out: Path) -> None:
    models = [m for m in MAIN if ((matrix.model == m) & (matrix.calib == "minmax") & (matrix.bit == MAIN[m])).any()]
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.0), squeeze=False)
    for ax, model in zip(axes[0], models):
        m = matrix[(matrix.model == model) & (matrix.calib == "minmax") & (matrix.bit == MAIN[model]) & (matrix.src_idx >= 0)]
        table = m.pivot(index="src_idx", columns="unit_idx", values="nmse")
        names = m.drop_duplicates("unit_idx").sort_values("unit_idx").unit.tolist()
        n = len(names)
        values = table.to_numpy().copy()
        values[np.tril_indices(n, -1)] = np.nan  # 뒤 단위에서 앞 단위로는 오차가 가지 않는다
        image = ax.imshow(values, cmap=SEQ, vmin=0, vmax=np.nanmax(values))
        for i in range(n):
            for j in range(i, n):
                dark = values[i, j] > 0.55 * np.nanmax(values)
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.6 if n > 10 else 6.5,
                        color="white" if dark else INK)
        ax.set_xticks(range(n), [short(u) for u in names[:-1]] + ["logits"], rotation=45, ha="right")
        ax.set_yticks(range(n), [short(u) for u in names])
        ax.set_xlabel("Where the error is measured")
        ax.set_ylabel("Quantized unit (all others FP32)")
        ax.set_title(f"{MODEL_NAMES[model]}, W{MAIN[model]}A{MAIN[model]}: NMSE vs FP32 at each position", loc="left")
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)
    fig.savefig(out / "fig6_propagation.png")
    plt.close(fig)


def fig_mechanism(units: pd.DataFrame, six: pd.DataFrame, out: Path) -> None:
    panels = []
    if not six.empty:
        g = six[six.condition == "W4A4"].sort_values("layer")
        panels.append(("6-Conv, W4A4 (5 seeds)", [f"C{i}" for i in g.layer], g.nmse_local, g.out_single, g.gain_pp))
    for model, bit in MAIN.items():
        g = units[(units.model == model) & (units.calib == "minmax") & (units.bit == bit)].sort_values("unit_idx")
        if len(g):
            panels.append((f"{MODEL_NAMES[model]}, W{bit}A{bit}", [short(u) for u in g.unit], g.nmse_local, g.out_single, g.gain_pp))
    rows = [("Error at the unit's own output\n(NMSE, only this unit quantized)", MUTED),
            ("Error it causes at the logits\n(NMSE, only this unit quantized)", BLUE),
            ("Accuracy gain when restored\nto FP32 (%p, all others quantized)", ORANGE)]
    widths = [max(len(p[1]), 6) for p in panels]
    fig, axes = plt.subplots(3, len(panels), figsize=(0.42 * sum(widths) + 2.2, 6.4), squeeze=False,
                             gridspec_kw={"width_ratios": widths})
    for c, (title, names, *series) in enumerate(panels):
        for r, (values, (label, color)) in enumerate(zip(series, rows)):
            ax = axes[r, c]
            ax.bar(range(len(names)), values, width=0.62, color=color)
            ax.axhline(0, color=MUTED, linewidth=0.6)
            ax.set_xticks(range(len(names)), names if r == 2 else [""] * len(names), rotation=45, ha="right")
            ax.margins(x=0.03)
            if c == 0:
                ax.set_ylabel(label)
            if r == 0:
                ax.set_title(title, loc="left")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    fig.savefig(out / "fig7_mechanism.png")
    plt.close(fig)


def fig_additivity(table: pd.DataFrame, out: Path) -> None:
    models = [m for m in MAIN if (table.model == m).any()]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        t = table[(table.model == model) & (table.family != "uniform")]
        low = t.bits.str.split().apply(lambda b: "4" in b)  # 4비트 단위가 하나라도 들어간 조합
        lo, hi = min(t.pred_nmse.min(), t.out_nmse.min()) * 0.7, max(t.pred_nmse.max(), t.out_nmse.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        for mask, color, label in ((~low, BLUE, "bits from {5, 6, 8}"), (low, ORANGE, "includes 4-bit units")):
            ax.scatter(t.pred_nmse[mask], t.out_nmse[mask], s=22, color=color, edgecolor="white", linewidth=0.6,
                       label=label, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.6)
        ax.set_xlabel("Predicted: sum of single-unit errors (logit NMSE)")
        ax.set_ylabel("Measured logit NMSE")
        ax.set_title(f"{MODEL_NAMES[model]}: {len(t)} random bit assignments", loc="left")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig8_additivity.png")
    plt.close(fig)


def fig_tokens(units: pd.DataFrame, out: Path) -> None:
    d = units[(units.model == "deit_tiny") & units.unit.str.startswith("block")]
    calibs = [c for c in ("minmax", "p99.99") if (d.calib == c).any()]
    if not calibs:
        return
    titles = {"minmax": "min-max range", "p99.99": "99.99th percentile range"}
    fig, axes = plt.subplots(1, len(calibs), figsize=(4.3 * len(calibs), 3.2), squeeze=False, sharey=True)
    for ax, calib in zip(axes[0], calibs):
        g = d[d.calib == calib].sort_values("unit_idx")
        x = np.arange(len(g))
        ax.plot(x, g.nmse_local_patch, color=BLUE, linewidth=2, marker="o", markersize=4, label="patch tokens (196)")
        ax.plot(x, g.nmse_local_cls, color=ORANGE, linewidth=2, marker="o", markersize=4, label="cls token (1)")
        ax.set_xticks(x, [short(u) for u in g.unit], rotation=45, ha="right")
        # 두 칸이 y축을 같이 쓰므로 범위를 전체 최댓값 기준으로 직접 정한다(자동 범위는 한쪽 칸 값에 맞춰져 잘림)
        ax.set_ylim(0, 1.08 * float(d[d.calib.isin(calibs)][["nmse_local_cls", "nmse_local_patch"]].max().max()))
        ax.set_xlabel("Quantized block (W5A5, all others FP32)")
        ax.set_title(f"DeiT-Tiny, {titles[calib]}", loc="left")
    axes[0, 0].set_ylabel("NMSE at the block's own output")
    axes[0, 0].legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "fig9_tokens.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/imagenet_propagation"))
    args = p.parse_args()
    summary, out = args.root / "summary", args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    six = pd.read_csv(summary / "sixconv.csv") if (summary / "sixconv.csv").exists() else pd.DataFrame()
    fig_propagation(pd.read_csv(summary / "matrix.csv"), out)
    units = pd.read_csv(summary / "units.csv")
    fig_mechanism(units, six, out)
    fig_tokens(units, out)
    if (summary / "additivity.csv").exists():
        fig_additivity(pd.read_csv(summary / "additivity.csv"), out)
    print("그림 저장:", sorted(f.name for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
