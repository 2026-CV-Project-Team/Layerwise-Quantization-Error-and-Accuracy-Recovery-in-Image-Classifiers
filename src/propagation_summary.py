"""오차 전달 분석 집계. exp_imagenet_propagation 결과(matrix.csv, evals.csv)를 읽는다.

출력 (results/imagenet_propagation/summary/)
- matrix.csv:        전달 행렬 원본을 모델별로 합친 것
- units.csv:         단위별 정리 (모델·보정·비트마다 한 줄씩)
    nmse_local   = 단위 i만 양자화했을 때 그 단위 출력의 오차           (행렬의 대각선)
    out_single   = 같은 상태의 최종 출력 오차                          (행렬의 마지막 열)
    amp          = out_single / nmse_local  (증폭률: i에서 생긴 상대 오차가 출력에서 몇 배가 되는가)
    nmse_all     = 전부 양자화한 상태에서 그 단위 출력의 오차 (누적 오차)
    own_share    = 누적 오차 중 그 단위에서 새로 생긴 몫 = 행렬[i, i] / Σ_k 행렬[k, i]
    add_ratio    = nmse_all / Σ_k 행렬[k, i]  (1이면 단위별 오차를 더한 값과 실제 누적 오차가 같음)
    gain_pp      = 전부 양자화 상태에서 단위 i만 FP32로 복원했을 때 정확도 회복 (rank 앞쪽 2,000장)
    out_gain     = 같은 복원에서 최종 출력 NMSE가 줄어든 양
    out_share    = out_single / Σ_k out_single_k  (단위별 최종 출력 오차의 합에서 차지하는 비율)
    *_cls, *_patch = DeiT 블록 출력을 cls 토큰과 patch 토큰으로 나눠 잰 nmse_local
- rank_corr.csv:     복원 이득 순위와 각 지표 순위의 Spearman ρ
- additivity.csv:    무작위 조합별 예측(단일 측정의 합)과 실제 최종 출력 오차
- additivity_family.csv: 비트 묶음별 요약 (실제 ÷ 예측의 중앙값, 예측과 실제의 순위 상관)
- split.csv:         W만, A만, 둘 다 양자화했을 때 단위별 최종 출력 오차
- calib.csv:         보정 방법별 전 단위 동일 비트 정확도와 최종 출력 오차
- sixconv.csv:       6-Conv 기존 결과(output_layers.csv)로 같은 지표를 계산한 것 (seed 평균)

주의
- 증폭률은 서로 다른 신호(단위 출력, logit)로 정규화한 상대 오차의 비율이다. 절대 오차의 배율이 아니다.
- 가산 예측은 "단위별 오차가 서로 독립이고 출력에서 그대로 더해진다"는 가정이다. 오차가 커지면 맞지 않는다.

사용 예:
    python -m src.propagation_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEYS = ["model", "calib", "bit"]


def rho(a, b) -> float:
    """Spearman ρ. 값이 전부 같아 순위를 매길 수 없으면 NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def unit_table(matrix: pd.DataFrame, evals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, calib, bit), m in matrix.groupby(KEYS, sort=False):
        n = m.unit_idx.max() + 1
        single = m[m.src_idx >= 0]
        table = single.pivot(index="src_idx", columns="unit_idx", values="nmse").to_numpy()  # [양자화한 단위, 잰 위치]
        all_row = m[m.src_idx < 0].sort_values("unit_idx")
        e = evals[(evals.model == model) & (evals.calib == calib) & (evals.bit == bit)]
        base = e[e.kind == "uniform"].iloc[0]
        restore = e[e.kind == "restore"].set_index("unit_idx")
        one = e[e.kind == "single_wa"].set_index("unit_idx")
        fp32 = evals[(evals.model == model) & (evals.kind == "fp32")].acc.iloc[0]
        out_single = table[:, n - 1]
        for i in range(n):
            diag = single[(single.src_idx == i) & (single.unit_idx == i)].iloc[0]
            column_sum = table[:, i].sum()  # 위치 i에서 잰 단위별 오차의 합 (i보다 뒤 단위는 0)
            rows.append({
                "model": model, "calib": calib, "bit": int(bit), "unit_idx": i, "unit": diag.unit,
                "nmse_local": table[i, i], "out_single": out_single[i], "amp": out_single[i] / table[i, i],
                "nmse_all": float(all_row.nmse.iloc[i]), "own_share": table[i, i] / column_sum,
                "add_ratio": float(all_row.nmse.iloc[i]) / column_sum,
                "out_share": out_single[i] / out_single.sum(),
                "acc_fp32": fp32, "acc_all": base.acc, "out_nmse_all": base.out_nmse,
                "drop_pp": fp32 - one.acc[i], "out_kl_single": one.out_kl[i],
                "gain_pp": restore.acc[i] - base.acc, "out_gain": base.out_nmse - restore.out_nmse[i],
                "out_kl_gain": base.out_kl - restore.out_kl[i],
                "nmse_local_cls": diag.nmse_cls, "nmse_local_patch": diag.nmse_patch,
            })
    return pd.DataFrame(rows)


def rank_corr(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in units.groupby(KEYS, sort=False):
        g = g.sort_values("unit_idx")
        row = dict(zip(KEYS, key), n_units=len(g), acc_all=g.acc_all.iloc[0],
                   top_gain=", ".join(g.sort_values("gain_pp", ascending=False).unit.head(3)),
                   top_nmse_all=", ".join(g.sort_values("nmse_all", ascending=False).unit.head(3)),
                   top_out_single=", ".join(g.sort_values("out_single", ascending=False).unit.head(3)))
        for col in ("nmse_all", "nmse_local", "amp", "out_single", "out_gain", "out_kl_gain", "drop_pp"):
            row[f"rho_gain_vs_{col}"] = rho(g.gain_pp, g[col])
        row["rho_gain_vs_depth"] = rho(g.gain_pp, g.unit_idx)
        rows.append(row)
    return pd.DataFrame(rows)


def additivity(evals: pd.DataFrame, params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """무작위 조합의 최종 출력 오차를 단일 측정의 합으로 예측해 실제와 비교한다."""
    rows = []
    for model, e in evals[evals.calib == "minmax"].groupby("model", sort=False):
        one = e[e.kind == "single_wa"]
        nmse = one.pivot(index="unit_idx", columns="bit", values="out_nmse")
        kl = one.pivot(index="unit_idx", columns="bit", values="out_kl")
        size = params[params.model == model].sort_values("unit_idx").params.to_numpy()
        targets = pd.concat([e[e.kind == "combo"], e[e.kind == "uniform"].assign(family="uniform")])
        for r in targets.itertuples():
            bits = [int(b) for b in r.w.split()]
            rows.append({
                "model": model, "family": r.family, "bits": r.w, "avg_bits": float((size * bits).sum() / size.sum()),
                "acc": r.acc, "out_nmse": r.out_nmse, "out_kl": r.out_kl,
                "pred_nmse": float(sum(nmse.loc[i, b] for i, b in enumerate(bits))),
                "pred_kl": float(sum(kl.loc[i, b] for i, b in enumerate(bits))),
            })
    table = pd.DataFrame(rows)
    table["ratio_nmse"] = table.out_nmse / table.pred_nmse
    table["ratio_kl"] = table.out_kl / table.pred_kl
    summary = []
    for (model, family), g in table.groupby(["model", "family"], sort=False):
        summary.append({
            "model": model, "family": family, "n": len(g),
            "out_nmse_median": g.out_nmse.median(), "ratio_nmse_median": g.ratio_nmse.median(),
            "ratio_nmse_min": g.ratio_nmse.min(), "ratio_nmse_max": g.ratio_nmse.max(),
            "ratio_kl_median": g.ratio_kl.median(),
            "rho_pred_vs_out_nmse": rho(g.pred_nmse, g.out_nmse), "rho_pred_kl_vs_out_kl": rho(g.pred_kl, g.out_kl),
            # 예측 오차가 클수록 정확도가 낮아야 하므로 음수가 정상
            "rho_pred_nmse_vs_acc": rho(g.pred_nmse, g.acc), "rho_pred_kl_vs_acc": rho(g.pred_kl, g.acc),
        })
    return table, pd.DataFrame(summary)


def split_table(evals: pd.DataFrame) -> pd.DataFrame:
    e = evals[(evals.calib == "minmax") & evals.kind.isin(["single_w", "single_a"])]
    rows = []
    for (model, bit), g in e.groupby(["model", "bit"], sort=False):
        both = evals[(evals.model == model) & (evals.calib == "minmax") & (evals.kind == "single_wa") & (evals.bit == bit)]
        both = both.set_index("unit_idx")
        w, a = (g[g.kind == k].set_index("unit_idx") for k in ("single_w", "single_a"))
        fp32 = evals[(evals.model == model) & (evals.kind == "fp32")].acc.iloc[0]
        for i in both.index:
            rows.append({"model": model, "bit": int(bit), "unit_idx": int(i), "unit": both.unit[i],
                         "out_w": w.out_nmse[i], "out_a": a.out_nmse[i], "out_wa": both.out_nmse[i],
                         "wa_over_sum": both.out_nmse[i] / (w.out_nmse[i] + a.out_nmse[i]),
                         "a_share": a.out_nmse[i] / (w.out_nmse[i] + a.out_nmse[i]),
                         "drop_w": fp32 - w.acc[i], "drop_a": fp32 - a.acc[i], "drop_wa": fp32 - both.acc[i]})
    return pd.DataFrame(rows)


def six_conv(exp: Path) -> pd.DataFrame:
    """6-Conv는 새로 돌리지 않고 기존 층별 결과로 같은 지표를 만든다 (세 과제 평균, seed 평균)."""
    path = exp / "summary" / "output_layers.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[d.task == "mean3"]
    g = d.groupby(["condition", "layer"], sort=False)[
        ["nmse_all", "nmse_only", "out_nmse_only", "out_nmse_all", "gain_pp", "drop_pp", "out_gain"]].mean().reset_index()
    g = g.rename(columns={"nmse_only": "nmse_local", "out_nmse_only": "out_single"})
    g["amp"] = g.out_single / g.nmse_local
    total = g.groupby("condition").out_single.transform("sum")
    g["out_share"] = g.out_single / total
    g["all_over_sum"] = g.out_nmse_all / total  # 전부 양자화한 실제 출력 오차 ÷ 단일 측정의 합
    return g


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/imagenet_propagation"))
    p.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    args = p.parse_args()
    summary = args.root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.concat(  # noqa: E731
        [pd.read_csv(f) for f in sorted(args.root.glob(f"*/{name}")) if f.parent.name != "summary"], ignore_index=True)
    matrix, evals, params = read("matrix.csv"), read("evals.csv"), read("units.csv")
    matrix.to_csv(summary / "matrix.csv", index=False)
    units = unit_table(matrix, evals)
    units.to_csv(summary / "units.csv", index=False)
    corr = rank_corr(units)
    corr.to_csv(summary / "rank_corr.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["unit", "nmse_local", "out_single", "amp", "nmse_all", "own_share", "add_ratio", "out_share", "gain_pp", "drop_pp"]
    for key, g in units.groupby(KEYS, sort=False):
        print("\n--", *key)
        print(g[show].round(3).to_string(index=False))
    print("\n[순위 상관: 복원 이득 vs 각 지표]")
    print(corr[KEYS + [c for c in corr.columns if c.startswith("rho_")]].round(2).to_string(index=False))

    calib = evals[evals.kind == "uniform"][["model", "calib", "bit", "acc", "out_nmse", "out_kl"]]
    calib.to_csv(summary / "calib.csv", index=False)
    print("\n[보정 방법별 전 단위 동일 비트 정확도 (%)]")
    print(calib.pivot_table(index=["model", "calib"], columns="bit", values="acc", sort=False).round(2).to_string())

    if (evals.kind == "single_w").any():
        split = split_table(evals)
        split.to_csv(summary / "split.csv", index=False)
        print("\n[W만 / A만 / 둘 다: 단위별 최종 출력 NMSE]")
        print(split.round(4).to_string(index=False))
    if (evals.kind == "combo").any():
        table, fam = additivity(evals, params)
        table.to_csv(summary / "additivity.csv", index=False)
        fam.to_csv(summary / "additivity_family.csv", index=False)
        print("\n[가산성: 비트 묶음별]")
        print(fam.round(3).to_string(index=False))
    six = six_conv(args.exp)
    if not six.empty:
        six.to_csv(summary / "sixconv.csv", index=False)
        print("\n[6-Conv, seed 평균]")
        print(six[six.condition.isin(["W4A4", "W5A5"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
