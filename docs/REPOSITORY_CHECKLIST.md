# Repository Release Checklist

Before public release:

- [ ] Confirm final manuscript title and author order.
- [ ] Confirm license with all authors and institutional policy.
- [ ] Replace placeholder repository URL in `CITATION.cff`.
- [ ] Add article DOI or Zenodo DOI after archive creation.
- [ ] Run `python -m compileall src scripts`.
- [ ] Re-run the dry-run data loader command.
- [ ] Verify that no raw data, checkpoints, tokens, or cloud artifacts are committed.
- [ ] Create a GitHub release or Zenodo archive after final code freeze.
