#!/usr/bin/env python3
"""
Optimized Iterative Boltzmann Inversion driver for ESPResSo 4.2.

This script orchestrates IBI iterations. It does NOT import espressomd itself;
it calls pypresso on run_espresso42_ibi_iteration.py, then calls
update_ibi_from_rdfs.py, checks convergence, and repeats.

It is meant to be run with normal Python, for example:

python run_ibi_optimized.py \
  --pypresso pypresso \
  --initial-potential "effective_potentials_ibi/initial_betaU/Rdfs/T - 0.2/rdf_rho_0.45_v_5.dat" \
  --target-rdf "effective_potentials_ibi/_extracted_target/Rdfs/T - 0.2/rdf_rho_0.45_v_5.dat" \
  --N 2000 --rho 0.45 --T 0.2 --dim 3 \
  --out ibi_runs/T0.2_rho0.45_v5 \
  --max-iter 10 --tol 1e-3 --alpha0 0.08
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(f"Command failed. See log: {log_path}", file=sys.stderr)
            try:
                lines = log_path.read_text(errors="replace").splitlines()
                print("\n--- tail of pypresso.log ---", file=sys.stderr)
                for line in lines[-80:]:
                    print(line, file=sys.stderr)
                print("--- end log tail ---\n", file=sys.stderr)
            except Exception as exc:
                print(f"Could not read log tail: {exc}", file=sys.stderr)
            raise subprocess.CalledProcessError(p.returncode, cmd)


def read_chi2(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    text = summary_path.read_text(errors="replace")
    m = re.search(r"chi2_rdf\s+([0-9eE+\-.]+)", text)
    return float(m.group(1)) if m else None


def adaptive_alpha(alpha0: float, iteration: int, chi2: float | None, prev_chi2: float | None) -> float:
    """Conservative alpha schedule for dense systems."""
    alpha = alpha0 / (1.0 + 0.15 * iteration)
    if chi2 is not None and prev_chi2 is not None and chi2 > 1.15 * prev_chi2:
        alpha *= 0.5
    return max(0.01, min(alpha, 0.2))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pypresso", default="pypresso")
    ap.add_argument("--initial-potential", required=True)
    ap.add_argument("--target-rdf", required=True)
    ap.add_argument("--out", required=True, help="Output directory for the whole IBI case.")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--dim", type=int, choices=[2, 3], default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-3, help="Stop when chi2_rdf is below this value.")
    ap.add_argument("--alpha0", type=float, default=0.08, help="Initial IBI mixing factor.")
    ap.add_argument("--warmup-steps", type=int, default=10000)
    ap.add_argument("--equil-steps", type=int, default=30000)
    ap.add_argument("--prod-blocks", type=int, default=60)
    ap.add_argument("--steps-per-block", type=int, default=1000)
    ap.add_argument("--rdf-bins", type=int, default=250)
    ap.add_argument("--force-clip", type=float, default=300.0)
    ap.add_argument("--min-dist", type=float, default=0.0)
    ap.add_argument("--init", choices=["lattice", "random"], default="lattice", help="Initial particle placement. Lattice is safer for tabulated/IBI potentials.")
    ap.add_argument("--tail-shift", action="store_true", default=True)
    ap.add_argument("--no-tail-shift", dest="tail_shift", action="store_false")
    ap.add_argument("--resume", action="store_true", help="Skip iterations that already have rdf_simulated.dat and potential_next.dat.")
    ap.add_argument("--log", action="store_true", help="Write each ESPResSo stdout/stderr to iter_XXX/pypresso.log.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    run_iter = here / "run_espresso42_ibi_iteration.py"
    updater = here / "update_ibi_from_rdfs.py"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    current_potential = Path(args.initial_potential)
    shutil.copy2(current_potential, out / "potential_iter000_input.dat")
    current_potential = out / "potential_iter000_input.dat"

    prev_chi2 = None
    history = []

    for it in range(args.max_iter):
        idir = out / f"iter_{it:03d}"
        idir.mkdir(parents=True, exist_ok=True)
        rdf_path = idir / "rdf_simulated.dat"
        summary_path = idir / "summary.txt"
        next_potential = out / f"potential_iter{it+1:03d}.dat"

        if not (args.resume and rdf_path.exists()):
            cmd = [
                args.pypresso, str(run_iter),
                "--potential", str(current_potential),
                "--target-rdf", args.target_rdf,
                "--N", str(args.N),
                "--rho", str(args.rho),
                "--T", str(args.T),
                "--dim", str(args.dim),
                "--seed", str(args.seed + it),
                "--warmup-steps", str(args.warmup_steps),
                "--equil-steps", str(args.equil_steps),
                "--prod-blocks", str(args.prod_blocks),
                "--steps-per-block", str(args.steps_per_block),
                "--rdf-bins", str(args.rdf_bins),
                "--force-clip", str(args.force_clip),
                "--min-dist", str(args.min_dist),
                "--init", args.init,
                "--out", str(idir),
            ]
            run(cmd, idir / "pypresso.log" if args.log else None)

        chi2 = read_chi2(summary_path)
        print(f"iteration {it}: chi2_rdf={chi2}")
        history.append((it, chi2))
        if chi2 is not None and chi2 <= args.tol:
            print(f"Converged at iteration {it}: chi2={chi2:.6e} <= tol={args.tol:.6e}")
            shutil.copy2(idir / "potential_used.dat", out / "potential_converged.dat")
            break

        alpha = adaptive_alpha(args.alpha0, it, chi2, prev_chi2)
        if not (args.resume and next_potential.exists()):
            cmd = [
                sys.executable, str(updater),
                "--potential", str(idir / "potential_used.dat"),
                "--target-rdf", args.target_rdf,
                "--sim-rdf", str(rdf_path),
                "--alpha", str(alpha),
                "--smooth-window", "7",
                "--out", str(next_potential),
            ]
            if args.tail_shift:
                cmd.append("--tail-shift")
            run(cmd)

        current_potential = next_potential
        prev_chi2 = chi2
    else:
        if current_potential.exists():
            shutil.copy2(current_potential, out / "potential_last.dat")

    with open(out / "history.dat", "w") as f:
        f.write("# iteration chi2_rdf\n")
        for it, chi2 in history:
            f.write(f"{it} {chi2 if chi2 is not None else float('nan'):.12e}\n")

    print(f"Wrote history: {out / 'history.dat'}")


if __name__ == "__main__":
    main()
