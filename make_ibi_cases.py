#!/usr/bin/env python3
"""Create a small case list for selected temperatures, densities and activities."""
import argparse
from pathlib import Path


def parse_list(s):
    return [x.strip() for x in s.replace(',', ' ').split() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ibi-root', default='effective_potentials_ibi')
    ap.add_argument('--temps', required=True, help='Example: "0.2 1.0"')
    ap.add_argument('--rhos', required=True, help='Example: "0.35 0.45"')
    ap.add_argument('--activities', required=True, help='Example: "0 2.5 5 10"')
    ap.add_argument('--out', default='ibi_cases.tsv')
    args = ap.parse_args()
    root = Path(args.ibi_root)
    lines = ['T\trho\tv\tinitial_potential\ttarget_rdf\tout_name']
    for T in parse_list(args.temps):
        Tdir = f'T - {T}'
        for rho in parse_list(args.rhos):
            for v in parse_list(args.activities):
                fname = f'rdf_rho_{rho}_v_{v}.dat'
                pot = root / 'initial_betaU' / 'Rdfs' / Tdir / fname
                rdf = root / '_extracted_target' / 'Rdfs' / Tdir / fname
                out_name = f'T{T}_rho{rho}_v{v}'.replace('.', 'p')
                if pot.exists() and rdf.exists():
                    lines.append(f'{T}\t{rho}\t{v}\t{pot}\t{rdf}\t{out_name}')
                else:
                    print(f'Missing: {pot} or {rdf}')
    Path(args.out).write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out} with {len(lines)-1} cases')

if __name__ == '__main__':
    main()
