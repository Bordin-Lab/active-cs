#!/usr/bin/env python3
"""
Run one IBI iteration in ESPResSo 4.2 from a tabulated betaU(r).

Purpose
-------
Given an IBI potential table (for example betaU_IBI_0 from the files generated
from your target RDFs), this script runs a passive equilibrium simulation with
that pair potential and writes the resulting RDF g_n(r).  You then update the
potential with update_ibi_from_rdfs.py and repeat.

Typical use
-----------
pypresso run_espresso42_ibi_iteration.py \
  --potential "effective_potentials_ibi/initial_betaU/Rdfs/T - 0.2/rdf_rho_0.45_v_5.dat" \
  --target-rdf "effective_potentials_ibi/_extracted_target/Rdfs/T - 0.2/rdf_rho_0.45_v_5.dat" \
  --N 2000 --rho 0.45 --T 0.2 --dim 3 \
  --out ibi_runs/rho_0.45_v_5_iter0

Notes
-----
- The potential table can have columns: r ... betaU ... . By default the script
  uses column index 3, consistent with the files I generated:
  r g_target_raw g_target_smooth betaU_IBI_0 U_IBI_0
- ESPResSo expects energy U(r), not betaU(r). The script converts using U=T*betaU
  with k_B=1.
- For dense systems, use several seeds and average the generated RDFs before
  updating the IBI potential.
"""

import argparse
import os
import sys
import math
import numpy as np

try:
    import espressomd
except Exception as exc:
    raise RuntimeError("This script must be run with ESPResSo/pypresso.") from exc


def read_table(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Could not read a numeric table with at least two columns: {path}")
    return data


def make_uniform_grid(r, betaU, n_grid=None):
    """Interpolate betaU to a uniform grid required by ESPResSo tabulated potentials."""
    order = np.argsort(r)
    r = np.asarray(r[order], dtype=float)
    betaU = np.asarray(betaU[order], dtype=float)

    mask = np.isfinite(r) & np.isfinite(betaU)
    r, betaU = r[mask], betaU[mask]
    unique = np.r_[True, np.diff(r) > 1e-12]
    r, betaU = r[unique], betaU[unique]

    if len(r) < 8:
        raise ValueError("Potential table has too few points.")

    if n_grid is None:
        n_grid = len(r)

    rmin = float(r[0])
    rmax = float(r[-1])
    ru = np.linspace(rmin, rmax, int(n_grid))
    bu = np.interp(ru, r, betaU)

    # Shift the tail to zero; this avoids a discontinuity at the cutoff.
    tail_n = max(5, min(25, len(bu) // 10))
    bu = bu - float(np.mean(bu[-tail_n:]))

    return ru, bu


def betaU_to_energy_force(r, betaU, T, force_clip=None):
    U = float(T) * betaU
    dUdr = np.gradient(U, r, edge_order=2)
    F = -dUdr
    # Force at the last grid point should go smoothly to zero at the cutoff.
    F[-1] = 0.0
    if force_clip is not None and force_clip > 0:
        F = np.clip(F, -force_clip, force_clip)
    return U, F


def setup_tabulated(system, r, U, F):
    """Set tabulated pair potential for type 0-0, compatible with ESPResSo 4.x."""
    rmin = float(r[0])
    rmax = float(r[-1])
    U = np.asarray(U, dtype=float)
    F = np.asarray(F, dtype=float)

    # Standard ESPResSo 4.x interface.
    try:
        system.non_bonded_inter[0, 0].tabulated.set_params(
            min=rmin, max=rmax, energy=U, force=F
        )
        return
    except Exception as first_exc:
        # Some builds expose the method but complain about arrays/lists.
        try:
            system.non_bonded_inter[0, 0].tabulated.set_params(
                min=rmin, max=rmax, energy=U.tolist(), force=F.tolist()
            )
            return
        except Exception as second_exc:
            msg = (
                "Could not set the tabulated potential. This ESPResSo build may have "
                "a different API. Tried system.non_bonded_inter[0,0].tabulated.set_params.\n"
                f"First error: {first_exc}\nSecond error: {second_exc}"
            )
            raise RuntimeError(msg)


def add_particles_lattice(system, N, box_l, dim):
    """Place particles on a regular lattice to avoid catastrophic overlaps."""
    if dim == 3:
        nside = int(np.ceil(N ** (1.0 / 3.0)))
        coords = (np.arange(nside) + 0.5) * box_l / nside
        count = 0
        for x in coords:
            for y in coords:
                for z in coords:
                    system.part.add(pos=[x, y, z], type=0)
                    count += 1
                    if count >= N:
                        return
    else:
        nside = int(np.ceil(np.sqrt(N)))
        coords = (np.arange(nside) + 0.5) * box_l / nside
        z = 0.5 * box_l
        count = 0
        for x in coords:
            for y in coords:
                system.part.add(pos=[x, y, z], type=0)
                count += 1
                if count >= N:
                    return


def add_particles_random(system, N, box_l, dim, seed, min_dist=0.0, max_attempts_per_particle=20000):
    rng = np.random.default_rng(seed)
    positions = []
    min_dist2 = min_dist * min_dist
    for i in range(N):
        accepted = False
        for _ in range(max_attempts_per_particle):
            pos = rng.random(3) * box_l
            if dim == 2:
                pos[2] = 0.5 * box_l
            if min_dist <= 0.0 or len(positions) == 0:
                accepted = True
            else:
                dr = np.asarray(positions) - pos
                dr -= box_l * np.rint(dr / box_l)
                if np.all(np.sum(dr * dr, axis=1) >= min_dist2):
                    accepted = True
            if accepted:
                system.part.add(pos=pos, type=0)
                positions.append(pos)
                break
        if not accepted:
            raise RuntimeError(
                f"Could not insert particle {i} with min_dist={min_dist}. "
                "Use a smaller --min-dist or initialize from a previous configuration."
            )


def compute_rdf_manual(system, r_min, r_max, n_bins, dim):
    """Compute RDF directly from particle positions.

    This avoids relying on system.analysis.rdf, which is absent in some
    ESPResSo 4.2 builds.  The normalization assumes a single-component
    system with periodic boundaries.  Pair distances are counted once
    (i<j) and then multiplied by 2 in the normalization.
    """
    pos = np.asarray(system.part.all().pos, dtype=float)[:, :dim]
    box = np.asarray(system.box_l, dtype=float)[:dim]
    N = pos.shape[0]
    if N < 2:
        raise RuntimeError("Need at least two particles to compute RDF.")

    max_allowed = 0.5 * float(np.min(box))
    if r_max > max_allowed:
        print(f"Warning: requested RDF r_max={r_max:.6g} exceeds L/2={max_allowed:.6g}; using L/2.")
        r_max = max_allowed
    if r_min < 0.0:
        r_min = 0.0
    if r_max <= r_min:
        raise RuntimeError(f"Invalid RDF range: r_min={r_min}, r_max={r_max}")

    edges = np.linspace(r_min, r_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=float)

    # Minimum-image pair distances.  For N~2000 this is fast enough and
    # avoids extra dependencies.
    for i in range(N - 1):
        dr = pos[i + 1:] - pos[i]
        dr -= box * np.rint(dr / box)
        dist = np.sqrt(np.sum(dr * dr, axis=1))
        h, _ = np.histogram(dist, bins=edges)
        hist += h

    centers = 0.5 * (edges[:-1] + edges[1:])
    volume = float(np.prod(box))
    rho = N / volume
    if dim == 3:
        shell = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    elif dim == 2:
        shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        rho = N / float(box[0] * box[1])
    else:
        raise RuntimeError("Only dim=2 or dim=3 are supported.")

    ideal_counts = 0.5 * N * rho * shell
    g = hist / np.maximum(ideal_counts, 1e-300)
    return centers, g


def compute_rdf(system, r_min, r_max, n_bins, dim):
    # Prefer ESPResSo's internal RDF when available; otherwise use the
    # manual implementation above.  Some ESPResSo 4.2 builds do not expose
    # system.analysis.rdf.
    if hasattr(system.analysis, "rdf"):
        try:
            r, rdf = system.analysis.rdf(r_min=r_min, r_max=r_max, r_bins=n_bins, types1=[0], types2=[0])
            return np.asarray(r), np.asarray(rdf)
        except TypeError:
            r, rdf = system.analysis.rdf(r_min=r_min, r_max=r_max, r_bins=n_bins, type_list_a=[0], type_list_b=[0])
            return np.asarray(r), np.asarray(rdf)
    return compute_rdf_manual(system, r_min, r_max, n_bins, dim)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--potential", required=True, help="Table containing r and betaU_n(r).")
    ap.add_argument("--target-rdf", default=None, help="Optional target RDF table; copied/interpolated to output for convenience.")
    ap.add_argument("--betaU-col", type=int, default=3, help="Zero-based column index for betaU in --potential.")
    ap.add_argument("--r-col", type=int, default=0, help="Zero-based column index for r in --potential.")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--rho", type=float, required=True, help="Number density. For dim=2 this is N/L^2; for dim=3 N/L^3.")
    ap.add_argument("--T", type=float, required=True, help="Simulation temperature, k_B=1.")
    ap.add_argument("--dim", type=int, choices=[2, 3], default=3)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-grid", type=int, default=0, help="Number of points for tabulated potential; 0 keeps input size.")
    ap.add_argument("--force-clip", type=float, default=500.0, help="Clip force table to avoid blowups; <=0 disables.")
    ap.add_argument("--min-dist", type=float, default=0.0, help="Optional rejection distance for random initial positions.")
    ap.add_argument("--init", choices=["lattice", "random"], default="lattice", help="Particle initialization. Lattice is safer for IBI/tabulated potentials.")
    ap.add_argument("--warmup-steps", type=int, default=20000)
    ap.add_argument("--equil-steps", type=int, default=50000)
    ap.add_argument("--prod-blocks", type=int, default=100)
    ap.add_argument("--steps-per-block", type=int, default=1000)
    ap.add_argument("--rdf-bins", type=int, default=250)
    ap.add_argument("--rdf-rmax", type=float, default=0.0, help="0 uses potential cutoff.")
    ap.add_argument("--out", required=True, help="Output prefix/directory for this run.")
    args = ap.parse_args()

    pdata = read_table(args.potential)
    r_raw = pdata[:, args.r_col]
    betaU_raw = pdata[:, args.betaU_col]
    r, betaU = make_uniform_grid(r_raw, betaU_raw, None if args.n_grid <= 0 else args.n_grid)
    U, F = betaU_to_energy_force(r, betaU, args.T, None if args.force_clip <= 0 else args.force_clip)

    if args.dim == 3:
        box_l = (args.N / args.rho) ** (1.0 / 3.0)
    else:
        box_l = math.sqrt(args.N / args.rho)

    system = espressomd.System(box_l=[box_l, box_l, box_l])
    system.time_step = args.dt
    system.cell_system.skin = 0.4
    if args.dim == 2:
        system.periodicity = [True, True, False]

    setup_tabulated(system, r, U, F)

    # Langevin thermostat. Different ESPResSo builds use either seed or rng_seed.
    try:
        system.thermostat.set_langevin(kT=args.T, gamma=args.gamma, seed=args.seed)
    except TypeError:
        system.thermostat.set_langevin(kT=args.T, gamma=args.gamma, rng_seed=args.seed)

    add_particles_lattice(system, args.N, box_l, args.dim) if args.init == "lattice" else add_particles_random(system, args.N, box_l, args.dim, args.seed, min_dist=args.min_dist)

    os.makedirs(args.out, exist_ok=True)
    np.savetxt(
        os.path.join(args.out, "potential_used.dat"),
        np.column_stack([r, betaU, U, F]),
        header="r betaU_used U_used force_used",
    )

    print(f"N={args.N} rho={args.rho} dim={args.dim} box_l={box_l:.8g}")
    print(f"Potential range: r=[{r[0]:.6g}, {r[-1]:.6g}], T={args.T}")
    sys.stdout.flush()

    if args.warmup_steps > 0:
        print(f"Warmup: {args.warmup_steps} steps")
        system.integrator.run(args.warmup_steps)
    if args.equil_steps > 0:
        print(f"Equilibration: {args.equil_steps} steps")
        system.integrator.run(args.equil_steps)

    rdf_rmax = args.rdf_rmax if args.rdf_rmax > 0 else float(r[-1])
    rdf_acc = None
    r_rdf = None
    print(f"Production: {args.prod_blocks} blocks x {args.steps_per_block} steps")
    for b in range(args.prod_blocks):
        system.integrator.run(args.steps_per_block)
        rr, gg = compute_rdf(system, r_min=float(r[0]), r_max=rdf_rmax, n_bins=args.rdf_bins, dim=args.dim)
        if rdf_acc is None:
            r_rdf = rr
            rdf_acc = np.zeros_like(gg, dtype=float)
        rdf_acc += gg
        if (b + 1) % max(1, args.prod_blocks // 10) == 0:
            print(f"  block {b+1}/{args.prod_blocks}")
            sys.stdout.flush()

    g_mean = rdf_acc / max(1, args.prod_blocks)
    np.savetxt(
        os.path.join(args.out, "rdf_simulated.dat"),
        np.column_stack([r_rdf, g_mean]),
        header="r g_simulated",
    )

    if args.target_rdf is not None:
        tdata = read_table(args.target_rdf)
        gt = np.interp(r_rdf, tdata[:, 0], tdata[:, 1])
        chi2 = float(np.trapz((g_mean - gt) ** 2, r_rdf) / (r_rdf[-1] - r_rdf[0]))
        np.savetxt(
            os.path.join(args.out, "rdf_compare.dat"),
            np.column_stack([r_rdf, gt, g_mean, g_mean - gt]),
            header="r g_target_interpolated g_simulated delta_g",
        )
        with open(os.path.join(args.out, "summary.txt"), "w") as f:
            f.write(f"N {args.N}\nrho {args.rho}\nT {args.T}\ndim {args.dim}\nbox_l {box_l}\n")
            f.write(f"chi2_rdf {chi2}\n")
        print(f"chi2_rdf={chi2:.8e}")

    print("Done.")


if __name__ == "__main__":
    main()
