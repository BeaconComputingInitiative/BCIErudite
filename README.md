<!-- BEACON COMPUTING INITIATIVE | Developer Seal: CJB-DEV-2026 | Chance Joaquin Boone -->
<!-- SHA-256 integrity record: docs/HYGIENE_SHA256SUMS -->

# BCIErudite
### Sovereign Language Model Research & Engineering

**Beacon Computing Initiative (BCI)** · **Developer Seal: CJB-DEV-2026**  
**Project class:** experimental neural-network / local language-model development  
**Repository:** https://github.com/BeaconComputingInitiative/BCIErudite  
**Status:** public research prototype; modular reconstruction is **proposed, not merged**

> **Mission:** Build an independently controlled, inspectable language-model stack from first principles—covering tokenization, neural architecture, training, checkpoints, evaluation, local inference, security boundaries, and operational evidence—without representing experimental behavior as production-ready intelligence.

**BCI principles:** Observe accurately · Preserve faithfully · Analyze transparently · Communicate securely · Act deliberately.

## Read this first: two different project states

The **`main` branch is the original prototype**. The more rigorous, unified neural-network reconstruction is in **[draft PR #2](https://github.com/BeaconComputingInitiative/BCIErudite/pull/2)**, which has not been merged. Do not assume paths or behavior described for that draft exist in the default checkout.

| Area | Default branch (`main`) | Proposed in draft PR #2 |
| --- | --- | --- |
| Model implementation | Multiple experimental neural paths needing integration review | One decoder-only Transformer core and native tokenizer |
| Training | Prototype training pipeline | Explicit causal LM training, evaluation and continuation |
| Checkpoints | Prototype-specific state | Native model/tokenizer identity, hashed weights and pointers |
| Inference | Streamlit chat prototype | Local streaming HTTP inference, separate from UI |
| User interface | Streamlit prototype plus historical PyQt variants | Thin Streamlit client; old UI versions preserved in `legacy/` |
| Evaluation | No demonstrated general language capability | 33 CPU tests and a synthetic end-to-end exercise reported in the PR |
| Datasets | Existing repository files retained | Existing datasets remain unchanged and are not automatically trained on |
| Readiness | Experimental / local-only | Engineering reconstruction under review; not a production assistant |

**No trained-model quality, production deployment, GPU parity, multilingual competence, or useful conversational performance is claimed for either branch.** Test results reported in a proposal are not evidence of deployment readiness.

## Why Erudite exists

Erudite is the BCI initiative for understanding and owning the **complete model lifecycle** instead of only wrapping a third-party inference service. The project is designed around four engineering commitments:

1. **Native development:** construct neural architectures and tokenization locally; no pretrained weights required by the proposed reconstruction.
2. **Reproducibility:** identify every configuration, tokenizer, model shape, checkpoint and evaluation boundary.
3. **Transparent failure:** return an explicit error when a contract fails instead of silently inventing a response.
4. **Data sovereignty:** retain local control over model artifacts and avoid automatic corpus uploads or unreviewed dataset modification.

## What is actually available on `main`

| Path | Responsibility |
| --- | --- |
| `app.py` | Experimental Streamlit dashboard with training and chat controls |
| `train_pipeline.py` | Prototype-specific training pipeline |
| `source/` | Separate tokenizer, model, interpreter and training implementations |
| `GUI/app.py` | Historical PyQt desktop controller prototype |
| `GUI/dashv1.py`, `GUI/dashv2.py` | Alternative UI experiments, not a declared supported interface |
| `dataset/` | Existing corpus files; do not modify as part of documentation or source cleanup |
| `tests/` | Prototype tests; passing them does not establish model usefulness |
| `docs/REPOSITORY_HYGIENE.md` | Historical repository findings and source-integrity controls |
| `docs/PORTFOLIO_AUDIT.md` | Branch/history snapshot and consolidation rationale |

### Explore the existing prototype

```bash
# BEACON COMPUTING INITIATIVE | Developer Seal: CJB-DEV-2026
git clone https://github.com/BeaconComputingInitiative/BCIErudite.git
cd BCIErudite
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

**Security boundary:** the original Streamlit prototype exposes shell-command and checkpoint-deletion controls. Run it only in an isolated, trusted **local development environment** after reviewing the source. Do not expose it to an untrusted network or treat it as a hardened service.

## The modular reconstruction under review

Draft PR #2 proposes one authoritative functional chain:

```text
Explicitly selected inputs
        |
Dataset/schema validation --> native byte BPE tokenizer
        |                            |
  Training + evaluation --> decoder-only Transformer
        |                   (causal attention, RoPE,
        |                    RMSNorm, SwiGLU, KV cache)
        v                            |
Immutable model weights + exact tokenizer/model metadata
        |                            |
  Native checkpoint pointer --> local streaming inference
                                      |
                             Thin Streamlit HTTP client
```

That branch documents model dimensions, masked causal loss, checkpoint recovery, bounded generation, Unicode decoding, input contracts and measured runtime metadata. It preserves original experimental source under `legacy/`, rather than deleting it. To inspect the proposal:

```bash
# BEACON COMPUTING INITIATIVE | Developer Seal: CJB-DEV-2026
git fetch origin codex/erudite-complete-project
git switch --track origin/codex/erudite-complete-project
```

Use the **README and instructions on that branch** for its installation and tests. Its starter architecture is intentionally small (approximately 0.92 million learned parameters; 256-token default context); a synthetic successful run proves software mechanics, not a capable assistant.

## Verification and acceptance

**Reported on the draft reconstruction branch:** 33 local CPU tests passed; synthetic data preparation, gradient training, checkpoint reconstruction and held-out evaluation were exercised. These are historical, branch-specific claims documented in [PR #2](https://github.com/BeaconComputingInitiative/BCIErudite/pull/2), not tests rerun by this README update.

**Not yet established:** full-corpus training and provenance clearance; useful task performance; device/GPU correctness and throughput; quantization; long-context stability; multi-user authentication; production deployment; operational recovery.

**Release only if and only if** the exact candidate commit passes its declared acceptance gates, source and artifact integrity checks, and relevant functional and security tests. Any remaining unverified claims must be explicitly labeled.

## Repository integrity and development conduct

All authored BCI documentation carries the project header, **Developer Seal: CJB-DEV-2026**, and exact-byte SHA-256 coverage in [docs/HYGIENE_SHA256SUMS](docs/HYGIENE_SHA256SUMS). A developer seal is attribution, not a digital signature. The manifest cannot include its own hash.

Do not rewrite historical commits merely for appearance. Keep datasets, credentials, generated weights, telemetry and local environments out of cosmetic cleanup. Maintain one authoritative contract per feature and review model changes in the sequence **input → tokenizer → tensors → loss → checkpoint → inference → UI**, with causal regression tests.

For current development proposals and evidence, see [draft PR #2](https://github.com/BeaconComputingInitiative/BCIErudite/pull/2) and the independent [repository audit PR #3](https://github.com/BeaconComputingInitiative/BCIErudite/pull/3). Neither is represented here as already merged.

---

© 2026 Beacon Computing Initiative · Developer Seal: CJB-DEV-2026. Repository visibility alone is not a license grant; verify ownership and distribution permissions before reuse.
