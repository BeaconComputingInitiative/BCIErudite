<!-- BEACON COMPUTING INITIATIVE | Project: BCIErudite | Developer Seal: CJB-DEV-2026 -->
<!-- File SHA-256: see SHA256SUMS -->

# BCIErudite — Sovereign Language Model Engineering

<p align="center">
  <img src="assets/bcierudite-icon.svg" width="224" alt="BCIErudite — neural knowledge emblem" />
</p>


**Beacon Computing Initiative**  
**Developer Seal: CJB-DEV-2026**  
**Project lead: Chance Joaquin Boone (CJB)**

Observe accurately. Preserve faithfully. Analyze transparently. Communicate securely. Act deliberately.

BCIErudite is an independently developed local language-model research project built from scratch: a native BPE tokenizer, a decoder-only Transformer, training and evaluation scripts, native checkpoint reconstruction, and a local streaming chat interface. It uses PyTorch primitives; no pretrained model weights, downloaded tokenizer, hosted inference endpoint, or LoRA base model is required.

**Project status:** a working, tested engineering foundation. The synthetic training-to-generation path is verified on CPU. Useful learned conversational capability and full-corpus training are not claimed. No trained weights are checked in.

## Engineering objectives and scope

BCIErudite exists to own the complete language-model lifecycle: **data contracts → tokenizer → tensors → training/evaluation → immutable checkpoints → local inference → observable UI**. In the reconstruction branch, all inference must use the same model/tokenizer/checkpoint contract produced by training; a missing or incompatible checkpoint must be reported, never substituted with a fabricated answer.

The project is deliberately **not a wrapper around Ollama, a hosted API, or a pretrained foundation model**. It does use established PyTorch tensor and autograd primitives. The starter network is intentionally small to make the mathematics, runtime, and regression tests inspectable before scaling.

### Evidence-driven capability matrix

| Dimension | Current evidence in this draft | Remaining acceptance work |
| --- | --- | --- |
| Native tokenizer / Transformer | CPU unit tests: round-trip, causality, caching, gradient flow | Larger context and hardware equivalence |
| Training / checkpointing | Real synthetic optimizer steps, native reload, held-out evaluation | Approved full-corpus integration, recovery testing |
| Streaming local inference | Unicode streams, context-limit and cancellation tests | Sustained throughput and latency measurements |
| Security boundaries | Loopback binding, Host/Origin checks and bounded requests | Authentication, threat assessment and deployment hardening |
| Useful conversational behavior | **Not demonstrated** by synthetic checks | Held-out real-world tasks and qualitative failure analysis |
| Reproducibility | Source SHA-256 manifest and checkpoint identity checks | Re-run gates against the exact release commit |

A historical validation report in this draft records **33 passing CPU tests on October 12, 2026**. This is evidence of the mechanics exercised in that environment, not a claim that the tests were rerun today. **No model weights are supplied.** Source hashes demonstrate byte integrity relative to a trusted manifest; the developer seal is attribution, not a cryptographic signature.

### Repository and release policy

- Preserve the original data and implementation history under documented boundaries; do not silently train on, rewrite, or publish the existing corpus.
- A release is acceptable **if and only if** its exact commit passes source integrity, reproducible model tests, explicit data-governance checks, measured performance targets and security criteria appropriate to its stated deployment scope.
- Record both **verified** and **unverified** behavior. Test success on a tiny synthetic language task does not establish language understanding, reliability, or general intelligence.
- Keep local-only services on loopback until authentication and exposure controls are separately designed and tested.

## Get started

Python 3.11+ is required. For a CPU installation:

```bash
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
git clone https://github.com/BeaconComputingInitiative/BCIErudite.git
cd BCIErudite
git switch --track origin/codex/erudite-complete-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/verify_hashes.py
python -m unittest discover -s tests -v
```

When reviewing this pull request, check out its branch before following these commands. For CUDA, use the official PyTorch installation appropriate to your hardware rather than the CPU wheel index. The dependency versions exercised are recorded in [validation notes](docs/VALIDATION.md).

## Run a complete synthetic demonstration

This exercises tokenizer fitting, real gradient training, checkpoint saving/reloading and held-out evaluation. It uses temporary synthetic inputs and does not read or change your existing datasets.

```bash
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
python scripts/smoke_run.py --output models/smoke --steps 30
python main.py --checkpoint models/smoke/best.json
```

In a second terminal inside the repository:

```bash
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
bash scripts/run_ui.sh
```

Open **http://127.0.0.1:8501**. The demo weights are a mechanics check; their output is not a reliable assistant. Its synthetic data is deleted when the command finishes. Repeating the demo requires a new output path to preserve previous snapshots.

## Project layout

| Location | Responsibility |
| --- | --- |
| `config/` | Runtime, guard/schema limits, and training configuration |
| `core/inference/model.py` | Erudite neural architecture, causal attention, RoPE, RMSNorm, SwiGLU, KV caching |
| `core/inference/tokenizer.py` | Native byte BPE, chat symbols, tokenizer fingerprint |
| `core/inference/checkpoint.py` | Model/tokenizer identity, immutable weight files, atomic pointers and hashes |
| `core/inference/engine.py` | Local generation, exact context budgeting, Unicode streaming |
| `core/guardrails/` | Input and dataset schema/resource contracts |
| `core/telemetry/` | Measured API metadata and host load; no conversation-text logging |
| `training/` | Explicit preparation, causal-LM training, evaluation and native continuation |
| `ui/components/` | Thin chat and measurement views; retrieval is visibly unimplemented |
| `scripts/` | Launchers, synthetic demo, integrity checking, optional user service/manual Git sync |
| `tests/` | Regression tests with real CPU tensor/training checks and isolated runtime checks |
| `docs/` | Architecture, API, validation, transparency and migration guidance |
| `legacy/` | Original model/tokenizer/UI source retained for reference, excluded from active imports/tests |
| `dataset/` | Existing repository datasets, preserved without edits |

Configuration files use `.yaml` names with JSON syntax, a valid YAML 1.2 subset parsed by the standard library. Do not add YAML-only comments/extensions inside those configuration files.

## Neural network

The starter configuration uses a 1,024-token vocabulary, width 128, four blocks, four heads, and 256-token context: **922,752 learned parameters**. Each block uses pre-RMSNorm attention with rotary query/key positions and a SwiGLU feed-forward branch, followed by residual additions. Input/output embedding weights are tied. The network initializes randomly.

Causal next-token loss ignores padding. Native continuation can optionally mask system/user targets for assistant-only training. Checkpoints embed architecture, tokenizer specification/fingerprint, training metadata and optimizer/RNG state. Inference reconstructs the exact model that training produced; old incompatible checkpoints are rejected.

## Existing data and real training

The existing dataset files are unchanged. The rebuild does not automatically migrate, rewrite, deduplicate or train on them. Preparation remains an **explicit, optional** operation writing to a separate new directory. Agree on data integration before using the main training configuration.

After you have supplied compatible prepared inputs, update `config/training_config.yaml`, choose a fresh run directory, and use:

```bash
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
python -m training.train_model
python -m training.evaluate_model --checkpoint models/runs/first/best.json
python main.py --checkpoint models/runs/first/best.json
```

[Training documentation](docs/TRAINING.md) explains the two objectives, exact identity checks, evaluation and native continuation. The test partition is reserved for final evaluation. Corpus quality, distribution permissions and learned-model quality require separate assessment.

## Validation and limits

CPU tests cover tokenizer round trips, causal masking, cached/full attention equivalence, gradients, tiny-sequence loss reduction, padding/assistant masks, checkpoint identity/integrity, continuation, exact-context rejection, Unicode streaming, HTTP failure recovery and thin UI rendering. [Executed validation](docs/VALIDATION.md) distinguishes those checks from untested GPU/performance/deployment behavior.

The starter architecture is a prototype, not a claim of strong reasoning or general intelligence. No vector retrieval, document-comprehension pipeline, pretrained adapters, GGUF integration or validated quantized serving is implemented. The local API binds loopback and has bounded generation/connections; it has no multi-user authentication. Training and inference should run separately to avoid hardware contention.

## Documentation

- [Architecture and core reconstruction](docs/ARCHITECTURE.md)
- [Detailed neural/core contracts](docs/CORE_RECONSTRUCTION.md)
- [Training and evaluation](docs/TRAINING.md)
- [Local API](docs/API_SPEC.md)
- [Migration from the original scripts](docs/MIGRATION.md)
- [Validation evidence](docs/VALIDATION.md)
- [Transparency and current limitations](docs/TRANSPARENCY.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Ownership and integrity

Copyright © 2026 Chance Joaquin Boone. [LICENSE](LICENSE) currently reserves rights; it is not an open-source license grant. Dependencies and preexisting datasets retain their own applicable terms. The developer seal is attribution, not a cryptographic signature. `SHA256SUMS` covers project source/documentation, excluding itself, original datasets and generated run artifacts. Checkpoint pointers separately verify weight hashes.
