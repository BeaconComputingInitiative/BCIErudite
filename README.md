<!-- BEACON COMPUTING INITIATIVE | Developer Seal: CJB-DEV-2026 | Chance Joaquin Boone -->
<!-- SHA-256: see docs/HYGIENE_SHA256SUMS -->

# BCIErudite

BCIErudite is the Beacon Computing Initiative's experimental language-model project. This repository is currently named `bci-sovereign-engine`.

The default branch contains the original prototype. The modular reconstruction is proposed in [draft PR #2](https://github.com/BeaconComputingInitiative/bci-sovereign-engine/pull/2); it is not yet part of this branch.

| Path | Role |
| --- | --- |
| `app.py` | Streamlit dashboard, training controls and chat prototype |
| `train_pipeline.py` | Training pipeline used by the Streamlit prototype |
| `source/` | Separate model, tokenizer, training and interpreter modules requiring compatibility review |
| `GUI/app.py` | PyQt desktop controller prototype |
| `GUI/dashv1.py`, `GUI/dashv2.py` | Historical dashboard alternatives; a supported choice is not yet established |
| `dataset/` | Existing corpus files; preserved without changes |

No trained model quality or production readiness is claimed. The Streamlit prototype includes shell-command execution and checkpoint deletion controls; evaluate it in a local development environment before use.

See [repository findings and cleanup rules](docs/REPOSITORY_HYGIENE.md). Review the modular reconstruction and its validation before choosing it as the canonical baseline.
