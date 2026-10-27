<!-- BEACON COMPUTING INITIATIVE | Developer Seal: CJB-DEV-2026 | Chance Joaquin Boone -->
<!-- SHA-256: see docs/HYGIENE_SHA256SUMS -->

# Repository review: bci-sovereign-engine

Reviewed: October 26, 2026. Default branch: `main`. Audited commit: `cf73e259d26978a3ffd8215d9a139ace9ec8a4f6`.

## Verified findings and follow-up

- No README exists on the default branch, so the project has no front door explaining the model, launch options, or limitations.
- `GUI/dashv1.py` and `GUI/dashv2.py` are competing numbered dashboards. Select a supported UI after tracing imports; do not simply select the larger filename.
- `app.py` and `train_pipeline.py` define their own transformer while `source/model.py` supplies another model path. A structural integration review is required before consolidating them.
- `dataset/dataset_combined_30k(1).jsonl` and `dataset/dataset_combined_30k.jsonl` have the same Git blob hash. Dataset contents and paths are preserved in this change.
- The reconstruction proposal is still draft PR #2, and the narrower naming proposal is draft PR #1. The default branch does not contain that reconstruction. Review #2 before retiring #1; neither was merged by this audit.
- Requested BCIErudite repository naming has not been applied; the current repository is still bci-sovereign-engine.

## Repository rules

- Maintain one documented current source tree and one supported starting point per product/platform. Give genuinely distinct tools purpose-based names.
- Use Git commits, branches and tags for revisions. Avoid final, latest, new, old, backup and copy suffixes on source files.
- Put downloadable builds and checksums in GitHub Releases. Keep recordings, telemetry, local environments and generated dependencies outside tracked source.
- Keep version-qualified protocol/API names and regression-test names when they describe an actual compatibility contract.
- Write commits as a concrete action and affected component, followed by the reason and relevant validation when needed. A late-night timestamp alone is not a quality defect.
- Release if and only if the documented acceptance gates pass on the exact release commit. Clearly label previews, recovered source, device validation gaps and historical releases.
- Keep the BCI header and developer seal on new authored files. Record their exact bytes in a SHA-256 manifest; do not attempt a self-referential file hash.

## Automated prevention

`Repository hygiene` runs on pushes and pull requests with read-only repository permissions. It inspects tracked files without executing project/application code. It rejects new confusing copy-style filenames, generated/runtime artifacts, recovery snapshot directories and byte-identical nonempty source copies. Existing paths are grandfathered against the audited commit, so this is prevention rather than a claim that historical issues are resolved. Empty package initializers are excluded from duplicate detection. Dataset duplicates are reported separately and left untouched.

Run locally: `python3 tools/repository_hygiene.py --baseline cf73e259d26978a3ffd8215d9a139ace9ec8a4f6` and `sha256sum -c docs/HYGIENE_SHA256SUMS`.

This workflow checks pushes and pull requests after integration, and it does not become a mandatory branch-protection rule automatically. It cannot prove semantic equivalence, runtime correctness, README accuracy or release readiness. Intentional shared code should be extracted into a common module; deliberate compatibility exceptions require a reviewed policy adjustment.

## Extended history audit

See [PORTFOLIO_AUDIT.md](PORTFOLIO_AUDIT.md) for the complete branch-history inventory, consolidation decisions and current review limits. The original findings above describe the pre-cleanup snapshot; the portfolio audit and README record the resulting state.
