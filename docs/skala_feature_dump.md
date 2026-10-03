# Native Skala feature capture

Set `CP2K_SKALA_FEATURE_DUMP` to an absolute output directory to retain the
protocol-2 input dictionaries of the last SCF energy evaluation. The SCF driver
arms the in-memory capture before initialization and each iteration. The first
actual model call replaces the previous evaluation; a cached KS update keeps
the last real input snapshot. This includes an initial XC evaluation from a
restart WFN. The driver writes it once when the SCF loop ends. No capture is
taken during the following force calculation.
Without this variable, no tensors are copied and no files are written.

This implementation is intended for single-point native Skala calculations on
one MPI rank. Multiple ranks are rejected when capture is requested. Reusing
the same directory replaces its manifest and metadata; use a separate directory
for each structure and model. Finite-difference XC diagnostics, outer constrained
DFT loops and molecular dynamics are outside the initially validated scope.

- `features.json`: schema, protocol, evaluation phase, batch filenames and the
  weighted XC integral calculated from each original model output.
- `batch_<n>_<key>.npy`: a detached, contiguous CPU copy of each exact dictionary
  value, in NPY v1 format. Names, dtypes and shapes are preserved, including the
  zero-length `atomic_grid_size_bound_shape` tensor. No pickle is used.
- `metadata.json`: revision, final SCF convergence flag, iteration count,
  tolerance, cell vectors in bohr, charge, multiplicity, spin channels, total
  energy, XC and non-XC subtotals, and the individual energy fields in hartree.
- `force_eval.inp`: expanded FORCE_EVAL input, including basis, potential, grid,
  XC, SCF and occupation settings. Referenced external files must also be pinned
  and hashed by the run record.
- `occupations.tsv`: final SCF occupation numbers and fractional k-point
  coordinates/weights. These are the final orbital snapshot; in unconverged
  or smeared calculations they need not describe the retained input density.

`scf_converged=false` is retained for diagnostics and must be rejected for
training. The captured density produced the recorded final SCF energy; the
subsequently written wavefunction can differ by the last SCF update even when
the convergence flag is true. A fixed-density restart check must establish its
own density agreement rather than assuming bitwise density identity.

For the native atom-composite GAPW-XC route, the entire Skala XC integral is in
`exc1`, while `exc` is zero. The reconstruction is
`non_xc_hartree + xc_hartree`, with `xc_hartree = exc + exc1`. Do not add a
second smooth-grid XC term or an additional one-centre XC correction. The
non-XC `hartree_1c` contribution remains included in `non_xc_hartree`.

The fork is based on `4d54163c07f27837ef60f601550c4261f6106950`.
Numerical acceptance, timings and the five-input run records are maintained in
the companion skala2pw fine-tuning plan; an implementation commit alone does
not certify acceptance.
