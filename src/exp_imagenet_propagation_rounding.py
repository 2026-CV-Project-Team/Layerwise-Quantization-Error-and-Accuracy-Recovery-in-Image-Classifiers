"""Compare layer-wise error propagation across round/trunc_zero/floor in W6A6.

This is a rank-split analysis only. It reuses the project's propagation Runner,
calibration split, model checkpoints, and output metrics; it does not use test data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp_imagenet import pick_device
from src.exp_imagenet_propagation import Runner
from src.imagenet_data import set_calib_seed
from src.quant import units as units_mod

BIT = 6
MODES = ("round", "trunc_zero", "floor")


def quant_fns(mode: str):
    def weight(weight, scale, q_max):
        x = weight / scale
        if mode == "round":
            q = torch.round(x)
        elif mode == "trunc_zero":
            q = torch.trunc(x)
        else:
            q = torch.floor(x)
        return scale * q.clamp(-q_max, q_max)

    def act(x, scale, zero, q_max):
        v = x / scale
        if mode == "round":
            q = torch.round(v)
        elif mode == "trunc_zero":
            q = torch.trunc(v)
        else:
            q = torch.floor(v)
        return scale * ((q + zero).clamp(0, q_max) - zero)

    return weight, act


class ProgressRunner(Runner):
    @torch.no_grad()
    def propagate(self, bit: int, mode: str) -> None:
        n, q = self.n, self.q
        states = [([bit] * n, [bit] * n)]
        for i in range(n):
            w, a = list(self.off), list(self.off)
            w[i], a[i] = bit, bit
            states.append((w, a))
        total = torch.zeros(len(states), n)
        token = torch.full((len(states), n, 2), float("nan"))
        outs: list[list[torch.Tensor]] = [[] for _ in states]
        n_batches = (len(self.x) + 99) // 100

        def nmse(ref: torch.Tensor, out: torch.Tensor) -> float:
            ref, out = ref.flatten(1), out.flatten(1)
            return float(((ref - out).pow(2).sum(1) / ref.pow(2).sum(1).clamp(min=1e-12)).sum())

        for batch_idx, start in enumerate(range(0, len(self.x), 100), start=1):
            batch = self.x[start : start + 100]
            from src.imagenet_data import to_input
            x = to_input(batch, self.device)
            q.apply(self.off, self.off)
            q.capture = []
            self.model(x)
            ref = q.capture
            for s, (w, a) in enumerate(states):
                q.apply(w, a)
                q.capture = []
                self.model(x)
                for k, (r, o) in enumerate(zip(ref, q.capture)):
                    total[s, k] += nmse(r, o)
                    if r.ndim == 3 and r.shape[1] == 197:
                        prev = torch.nan_to_num(token[s, k])
                        token[s, k] = prev + torch.tensor([
                            nmse(r[:, :1], o[:, :1]), nmse(r[:, 1:], o[:, 1:])
                        ])
                outs[s].append(q.capture[-1].float().cpu())
            if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == n_batches:
                print(f"{self.name} {mode} W{bit}A{bit}: propagation batch {batch_idx}/{n_batches}", flush=True)

        q.capture = None
        q.apply(self.off, self.off)
        total, token = total / len(self.x), token / len(self.x)
        for s, (w, a) in enumerate(states):
            src = s - 1
            out = torch.cat(outs[s])
            if src < 0:
                self.record("uniform_m", w, a, out, bit=bit)
            else:
                self.record("single_wa", w, a, out, bit=bit, unit_idx=src, unit=self.units[src].name)
            for k, unit in enumerate(self.units):
                self.matrix.append({
                    "model": self.name, "calib": self.calib, "mode": mode, "bit": bit,
                    "src_idx": src, "src": "all" if src < 0 else self.units[src].name,
                    "unit_idx": k, "unit": unit.name, "nmse": float(total[s, k]),
                    "nmse_cls": float(token[s, k, 0]), "nmse_patch": float(token[s, k, 1]),
                })


def run_one(name: str, mode: str, n_images: int, out_root: Path, force: bool) -> None:
    out_dir = out_root / name / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / "complete.json"
    if complete.exists() and not force:
        print(f"skip completed {name}/{mode}; use --force to rerun", flush=True)
        return

    set_calib_seed(0)
    old_weight, old_act = units_mod.fake_quant_weight, units_mod.fake_quant_act
    units_mod.fake_quant_weight, units_mod.fake_quant_act = quant_fns(mode)
    t0 = time.time()
    runner = None
    try:
        print(f"start {name}/{mode}: W6A6, rank_images={n_images}, device={pick_device()}", flush=True)
        runner = ProgressRunner(name, pick_device(), n_images=n_images)
        runner.calib = "seed0_minmax"
        runner.q.calibrate(runner.calib_batches())
        runner.ref = runner.logits(runner.off, runner.off)
        runner.record("fp32", runner.off, runner.off, runner.ref)
        runner.evaluate("uniform_m", [BIT] * runner.n, [BIT] * runner.n, bit=BIT)
        runner.propagate(BIT, mode)

        all_w, all_a = [BIT] * runner.n, [BIT] * runner.n
        for i, unit in enumerate(runner.units):
            w, a = list(all_w), list(all_a)
            w[i] = a[i] = None
            runner.evaluate("restore", w, a, bit=BIT, unit_idx=i, unit=unit.name)
        for i, unit in enumerate(runner.units):
            w, a = list(runner.off), list(runner.off)
            w[i] = BIT
            runner.evaluate("single_w", w, a, bit=BIT, unit_idx=i, unit=unit.name)
            w, a = list(runner.off), list(runner.off)
            a[i] = BIT
            runner.evaluate("single_a", w, a, bit=BIT, unit_idx=i, unit=unit.name)
            if i == runner.n - 1 or (i + 1) % 4 == 0:
                print(f"{name} {mode}: restore and W/A split {i + 1}/{runner.n}", flush=True)
        runner.save(out_dir)
        (out_dir / "complete.json").write_text(json.dumps({
            "model": name, "mode": mode, "bit": BIT, "calibration_seed": 0,
            "rank_images": n_images, "device": str(pick_device()),
            "elapsed_seconds": round(time.time() - t0, 2), "test_evaluated": False,
        }, indent=2), encoding="utf-8")
        print(f"done {name}/{mode}: {time.time() - t0:.1f}s", flush=True)
    finally:
        if runner is not None:
            try:
                runner.q.remove()
            except Exception:
                pass
        units_mod.fake_quant_weight, units_mod.fake_quant_act = old_weight, old_act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=("resnet18", "deit_tiny"), default=("resnet18", "deit_tiny"))
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="results/imagenet_propagation_rounding")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"device={pick_device()} modes={args.modes} rank_images={args.n}; test split will not be used", flush=True)
    for name in args.models:
        for mode in args.modes:
            run_one(name, mode, args.n, out_root, args.force)


if __name__ == "__main__":
    main()
