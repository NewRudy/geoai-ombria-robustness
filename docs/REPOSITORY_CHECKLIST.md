# Repository Release Checklist

Current code-release state:

- [x] Repository contains code only; manuscript drafts and private metadata are excluded.
- [x] Public repository URL is recorded in `CITATION.cff`.
- [x] MIT license file is present.
- [x] Confirmatory protocol and evidence boundaries are explicit.
- [x] `python scripts/audit_repository.py` passes.
- [x] Python and shell syntax checks pass.
- [x] No raw data, checkpoints, tokens, local paths, or generated ZIP archives are tracked.
- [x] OMBRIA upstream revision is pinned.
- [x] Kaggle smoke and full notebooks use the immutable release tag.
- [x] P100 compatibility is checked with an architecture probe and real CUDA convolution.
- [x] Kaggle clone setup is safe when the notebook is rerun in the same kernel.
- [x] Returned archives retain training trajectories, runtime provenance, console logs, file hashes, and checkpoint hashes linked to every evaluation configuration.
- [x] Three-seed 95% intervals use the Student-t critical value for df = 2; one-seed Smoke intervals are explicitly not estimable.

Publication metadata still pending:

- [ ] Confirm the final manuscript title and author order before article submission.
- [ ] Add article DOI or Zenodo DOI after archive creation.
- [ ] Add the final journal citation after acceptance.
