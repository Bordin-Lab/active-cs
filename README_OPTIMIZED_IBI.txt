Optimized IBI workflow for ESPResSo 4.2
=======================================

Goal
----
Reduce manual work and avoid over-iterating. The driver runs one ESPResSo
simulation, compares the generated RDF with the target RDF, updates the
potential with adaptive alpha, and stops automatically when chi2_rdf < tol.

Recommended first run
---------------------
Use only a few representative cases first, for example the highest density and
three activities. Use smaller N for debugging.

1) Create a case list:

python make_ibi_cases.py \
  --ibi-root effective_potentials_ibi \
  --temps "0.2" \
  --rhos "0.45" \
  --activities "0 2.5 5 10" \
  --out ibi_cases.tsv

2) Run cases:

OUTROOT=ibi_runs N=2000 DIM=3 MAX_ITER=8 TOL=1e-3 ALPHA0=0.08 \
  ./run_cases_from_list.sh ibi_cases.tsv

For dense systems
-----------------
Start with conservative values:
  ALPHA0=0.05 to 0.08
  TOL=1e-3 for quantitative comparison, 1e-2 for qualitative comparison
  MAX_ITER=5 to 10 initially

Practical convergence criterion
-------------------------------
Inspect each case's:
  ibi_runs/<case>/history.dat
  ibi_runs/<case>/iter_XXX/rdf_compare.dat

If chi2_rdf decreases and then plateaus, stop. For an article focused on
structural trends, it is usually better to report that the potential was taken
after a fixed small number of iterations or at convergence, and to show that
RDFs are reproduced within the chosen tolerance.

Important interpretation
------------------------
For active systems, the IBI potential is an activity-dependent structural
effective potential: it reproduces g(r), but it is not a unique equilibrium
thermodynamic pair potential and should not be expected to reproduce dynamics,
pressure or response functions.
