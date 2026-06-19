#!/usr/bin/env python3
"""
Update an IBI potential using target and simulated RDFs.

Formula:
  betaU_{n+1}(r) = betaU_n(r) + alpha * ln[g_sim(r)/g_target(r)]

Example:
python update_ibi_from_rdfs.py \
  --potential ibi_runs/rho_0.45_v_5_iter0/potential_used.dat \
  --target-rdf effective_potentials_ibi/_extracted_target/Rdfs/T\\ -\\ 0.2/rdf_rho_0.45_v_5.dat \
  --sim-rdf ibi_runs/rho_0.45_v_5_iter0/rdf_simulated.dat \
  --alpha 0.05 \
  --out ibi_runs/rho_0.45_v_5_iter1/potential_iter1.dat
"""

import argparse
import os
import numpy as np


def load(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise ValueError(f"Invalid table: {path}")
    return data


def smooth_moving(y, window):
    if window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yy = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(yy, kernel, mode="valid")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--potential", required=True, help="Previous potential table. Usually potential_used.dat from the ESPResSo run.")
    ap.add_argument("--target-rdf", required=True, help="Target RDF table, columns r g_target.")
    ap.add_argument("--sim-rdf", required=True, help="Simulated RDF table, columns r g_sim.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--r-col", type=int, default=0)
    ap.add_argument("--betaU-col", type=int, default=1, help="Column containing betaU_n in --potential. For potential_used.dat this is 1.")
    ap.add_argument("--g-col-target", type=int, default=1)
    ap.add_argument("--g-col-sim", type=int, default=1)
    ap.add_argument("--g-floor", type=float, default=1e-5)
    ap.add_argument("--smooth-window", type=int, default=5, help="Moving-average smoothing window for the correction.")
    ap.add_argument("--tail-shift", action="store_true", help="Shift betaU tail average to zero after update.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = load(args.potential)
    t = load(args.target_rdf)
    s = load(args.sim_rdf)

    r = p[:, args.r_col]
    betaU = p[:, args.betaU_col]

    gt = np.interp(r, t[:, 0], t[:, args.g_col_target])
    gs = np.interp(r, s[:, 0], s[:, args.g_col_sim])
    gt = np.maximum(gt, args.g_floor)
    gs = np.maximum(gs, args.g_floor)

    correction = args.alpha * np.log(gs / gt)
    correction = smooth_moving(correction, args.smooth_window)
    betaU_new = betaU + correction

    if args.tail_shift:
        n_tail = max(5, min(25, len(betaU_new) // 10))
        betaU_new = betaU_new - np.mean(betaU_new[-n_tail:])

    chi2_before = float(np.trapz((gs - gt) ** 2, r) / (r[-1] - r[0]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savetxt(
        args.out,
        np.column_stack([r, betaU_new, betaU, correction, gt, gs]),
        header="r betaU_new betaU_old delta_betaU g_target g_simulated",
    )
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as f:
        f.write(f"alpha {args.alpha}\n")
        f.write(f"chi2_rdf {chi2_before}\n")
        f.write(f"max_abs_delta_betaU {np.max(np.abs(correction))}\n")
    print(f"Wrote {args.out}")
    print(f"chi2_rdf={chi2_before:.8e}, max|delta_betaU|={np.max(np.abs(correction)):.6g}")


if __name__ == "__main__":
    main()
