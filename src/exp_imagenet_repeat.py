"""ImageNet 실험 반복 측정: 보정(calibration) 세트를 바꿔 같은 실험을 다시 하고 seed 간 편차를 집계한다.

왜 보정 세트를 바꾸나
- 6-Conv는 학습 seed 5개로 반복했지만, ImageNet 모델은 사전학습 가중치 그대로라 학습 seed가 없다.
- 이 설정에서 실행마다 달라질 수 있는 것은 활성값 범위를 정하는 보정 이미지 2,048장뿐이다.
  min-max 보정은 이미지 몇 장의 극단값에 범위가 좌우되므로, 기존 1회 결과(ρ, 정책 순위)가
  보정 세트를 바꿔도 유지되는지 확인해야 한다.
- rank 5,000장, test 10,000장은 바꾸지 않는다(평가 이미지까지 바꾸면 원인이 둘이 됨).

잠정 가정 (팀 확인 필요)
- 보정 세트 seed 0(기존) + 1~4 = 5개. 6-Conv의 seed 5개와 수를 맞춘 것.
- 각 seed에서 기존 1회 측정과 똑같은 단계를 그대로 돌린다:
  조건별 정확도 -> 단위별 복원·교체 -> 최종 출력 오차 -> 상위 k 승격 정책 비교.
  정책의 순위도 그 seed의 rank 결과로만 정한다(seed 0의 순위를 다른 seed에 재사용하지 않음).
- 단위별 조건은 기존과 같다: ResNet-18 W4A4·W5A5, DeiT-Tiny W4A4·W5A5·W6A6.
  승격 기준 비트도 기존과 같다: ResNet-18 W4A4·W5A5, DeiT-Tiny W5A5·W6A6.

결과 위치
- seed 0: results/imagenet/ (기존 결과 그대로)
- seed s: results/imagenet_calib/s{s}/ (폴더 구조는 results/imagenet/과 같음)
- 집계:   results/imagenet_calib/summary/
    conditions_repeat.csv  조건별 정확도의 seed 평균·표준편차·최소·최대
    units_repeat.csv       단위별 gain_pp·drop_pp·nmse_all 등의 seed 평균·표준편차
    rank_corr_seeds.csv    seed마다 ρ(누적 NMSE vs 복원 이득)와 이득 1위 단위
    rank_corr_repeat.csv   위의 seed 평균·표준편차, 이득 1위 단위가 같았던 seed 수
    mixed_topk_seeds.csv   seed마다 정책별 상위 k 승격 정확도
    mixed_topk_repeat.csv  위의 seed 평균·표준편차

실행 예 (seed 하나에 2시간 안팎 추정, M4 MPS 기준)
    python -m src.exp_imagenet_repeat --seeds 1 2 3 4      # 받기 -> 전처리 -> 실험 -> 집계
    python -m src.exp_imagenet_repeat --summary-only        # 집계만 다시
보정 이미지를 받으려면 Hugging Face 로그인이 필요하다(README의 ImageNet 데이터 절 참고).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "imagenet"  # seed 0
REPEAT = ROOT / "results" / "imagenet_calib"
UNIT_CONDITIONS = {"resnet18": ["W4A4", "W5A5"], "deit_tiny": ["W4A4", "W5A5", "W6A6"]}
MIXED_BASES = {"resnet18": ["W4A4", "W5A5"], "deit_tiny": ["W5A5", "W6A6"]}


def seed_root(seed: int) -> Path:
    return BASE if seed == 0 else REPEAT / f"s{seed}"


def step(args: list[str]) -> None:
    """같은 파이썬으로 모듈 하나를 실행. 실패하면 바로 멈춘다(뒤 단계가 앞 결과를 읽으므로)."""
    t0 = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)
    print(f"<< {time.time() - t0:.0f}s", flush=True)


def run_seed(seed: int, keep_jpeg: bool) -> None:
    assert seed > 0, "seed 0은 기존 결과(results/imagenet)를 쓴다"
    tag = ["--calib-seed", str(seed)]
    out = ["--out", str(seed_root(seed).relative_to(ROOT))]
    cache = ROOT / "data" / "imagenet" / "cache" / f"deit_tiny_calib_s{seed}_x.npy"
    if not cache.exists():
        step(["src.imagenet_data", "--calib", "--cache", *tag])
        if not keep_jpeg:  # 디스크 여유가 적어서 전처리 배열만 남긴다(목록 json이 있어 다시 받을 수 있음)
            shutil.rmtree(ROOT / "data" / "imagenet" / "calib" / f"s{seed}", ignore_errors=True)
    step(["src.exp_imagenet", *tag, *out])
    for model, conds in UNIT_CONDITIONS.items():
        step(["src.exp_imagenet_units", "--models", model, "--conditions", *conds, *tag, *out])
    step(["src.exp_imagenet_output", *tag, *out])
    # output_summary.imagenet()만 부른다(main은 6-Conv 집계도 다시 쓰므로 피함)
    step(["src.exp_imagenet_repeat", "--output-summary", str(seed_root(seed))])
    for model, bases in MIXED_BASES.items():
        step(["src.exp_imagenet_mixed", "--models", model, "--bases", *bases, *tag, *out])


def gather(name: str, seeds: list[int]) -> pd.DataFrame:
    """seed별 summary/{name}을 모아 calib_seed 열을 붙인다. 없는 seed는 건너뛴다."""
    parts = []
    for s in seeds:
        path = seed_root(s) / "summary" / name
        if path.exists():
            parts.append(pd.read_csv(path).assign(calib_seed=s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def spread(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """keys별로 cols의 seed 평균·표준편차(·최소·최대). n_seeds로 몇 개 seed가 모였는지 같이 남긴다."""
    g = df.groupby(keys, sort=False)
    out = g.size().rename("n_seeds").to_frame()
    for c in cols:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_sd"] = g[c].std()  # seed 1개면 NaN
        out[f"{c}_min"] = g[c].min()
        out[f"{c}_max"] = g[c].max()
    return out.reset_index()


def summarize(seeds: list[int]) -> None:
    dst = REPEAT / "summary"
    dst.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40)

    cond = gather("conditions.csv", seeds)
    if not cond.empty:
        rep = spread(cond, ["model", "set", "condition"], ["acc"])
        rep.to_csv(dst / "conditions_repeat.csv", index=False)
        print("\n[조건별 정확도, test] seed 평균 ± 표준편차")
        print(rep[rep.set == "test"][["model", "condition", "n_seeds", "acc_mean", "acc_sd", "acc_min", "acc_max"]]
              .round(2).to_string(index=False))

    units = gather("units.csv", seeds)
    if not units.empty:
        cols = ["acc_all", "gain_pp", "drop_pp", "nmse_all", "nmse_only"]
        out_units = gather("output_units.csv", seeds)
        if not out_units.empty:  # 최종 출력 오차 기준(out, out_kl 정책용)도 같이 본다
            units = units.merge(out_units[["model", "condition", "unit", "calib_seed", "out_gain", "out_kl_gain"]],
                                on=["model", "condition", "unit", "calib_seed"], how="left")
            cols += ["out_gain", "out_kl_gain"]
        spread(units, ["model", "condition", "unit"], cols).to_csv(dst / "units_repeat.csv", index=False)

    corr = gather("rank_corr.csv", seeds)
    if not corr.empty:
        keep = ["model", "condition", "calib_seed", "acc_all", "rho_nmse_all_vs_gain", "p_nmse_all_vs_gain",
                "rho_drop_pp_vs_gain", "top3_overlap_nmse_all", "top_gain", "top_nmse_all"]
        corr = corr[keep]
        corr = corr.assign(top1_gain=corr.top_gain.str.split(", ").str[0])
        corr.to_csv(dst / "rank_corr_seeds.csv", index=False)
        rep = spread(corr, ["model", "condition"], ["rho_nmse_all_vs_gain", "rho_drop_pp_vs_gain", "top3_overlap_nmse_all"])
        # 이득 1위 단위가 seed 사이에 얼마나 일정한가: 가장 흔한 단위와 그 횟수
        mode = corr.groupby(["model", "condition"], sort=False).top1_gain.agg(
            lambda s: f"{s.mode().iloc[0]} ({(s == s.mode().iloc[0]).sum()}/{len(s)})")
        rep = rep.merge(mode.rename("top1_gain_mode").reset_index(), on=["model", "condition"])
        rep.to_csv(dst / "rank_corr_repeat.csv", index=False)
        print("\n[ρ(누적 NMSE vs 복원 이득)] seed 평균 ± 표준편차")
        print(rep[["model", "condition", "n_seeds", "rho_nmse_all_vs_gain_mean", "rho_nmse_all_vs_gain_sd",
                   "rho_nmse_all_vs_gain_min", "rho_nmse_all_vs_gain_max", "top1_gain_mode"]].round(3).to_string(index=False))

    mix = gather("mixed_topk.csv", seeds)
    if not mix.empty:
        # random은 각 seed 안에서 이미 무작위 5개 평균이다. 여기서는 보정 seed 간 편차를 본다
        mix = mix.drop(columns=["sd_test", "n"], errors="ignore")
        mix.to_csv(dst / "mixed_topk_seeds.csv", index=False)
        rep = spread(mix, ["model", "base", "policy", "k"], ["acc_test", "avg_bits"])
        rep.to_csv(dst / "mixed_topk_repeat.csv", index=False)
        print("\n[상위 k=3 승격, test 정확도] seed 평균 ± 표준편차")
        view = rep[rep.k == 3][["model", "base", "policy", "n_seeds", "acc_test_mean", "acc_test_sd", "avg_bits_mean"]]
        print(view.sort_values(["model", "base", "acc_test_mean"], ascending=[True, True, False]).round(2).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4], help="새로 돌릴 보정 seed (1 이상)")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--keep-jpeg", action="store_true", help="보정 JPEG를 지우지 않고 남김")
    p.add_argument("--output-summary", type=Path, help=argparse.SUPPRESS)  # 내부용
    args = p.parse_args()
    if args.output_summary:
        from .output_summary import imagenet
        imagenet(args.output_summary)
        return
    if not args.summary_only:
        for s in args.seeds:
            run_seed(s, args.keep_jpeg)
            summarize([0, *range(1, s + 1)])  # seed 하나 끝날 때마다 중간 집계
    found = [0] + sorted(int(d.name[1:]) for d in REPEAT.glob("s*") if d.name[1:].isdigit())
    summarize(found)


if __name__ == "__main__":
    main()
