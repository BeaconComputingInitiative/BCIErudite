# Erudite — Rebuild Transparency

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

## Scope and user constraints

This is a source-only reconstruction of the neural network and core scripts. The project retains the core/training/ui/scripts/config/docs framework. Dataset changes were explicitly frozen: no original dataset contents were modified, no new datasets are included in this source change, and no further project-corpus preparation or training was performed after that instruction.

The preserved old neural and UI source is under `legacy/`. Active modules live under `core/`, `training/`, and the thin `ui/` package. This source-only change is presented on a reviewable GitHub branch; it preserves existing dataset blobs and excludes private-derived audit statistics/hashes.

## Implemented and exercised

- Native byte BPE with reserved chat symbols, reproducible merge selection, actual vocabulary sizing, strict unknown-token rejection, and tokenizer fingerprinting.
- Randomly initialized EruditeLM using RoPE, RMSNorm, SwiGLU, causal SDPA, residual paths, tied input/output weights, and native key/value caching.
- Shifted causal LM loss, ignored padding, optional assistant-only masks, finite loss/gradient checks, AdamW and warmup/cosine learning-rate scheduling.
- Exact native checkpoint reconstruction, tokenizer/dataset identity contracts, immutable step files and atomic checkpoint pointers.
- Held-out token-weighted evaluation, explicit continuation into a fresh run directory, CPU device placement, incremental Unicode streaming and context-overflow rejection.
- A synthetic end-to-end exercise that prepares temporary inputs, performs real gradient training, saves/reloads model+tokenizer, evaluates held-out targets, generates locally, changes pointer revisions and continues training.

See VALIDATION.md for the actual environment/checks. Tiny synthetic examples establish mechanics, not useful language skill. No trained weights are distributed.

## Deliberately unclaimed

No full-corpus training success, conversational competence, source/dataset quality improvement, semantic deduplication, GPU correctness/throughput, long-context performance, power-loss durability, quantized serving, GGUF support, document retrieval, or multi-user deployment is claimed.

Existing datasets were not migrated or trained on for this source reconstruction. No full-corpus training success is claimed.

The active runtime has no shell terminal or scripted boot/telemetry claims. UI metadata comes from measured API timings and OS load. Schema/size guards do not constitute a semantic moderation classifier or a cryptographic trust system. Checkpoint hashes detect byte changes relative to a trusted pointer; they do not establish authenticity.

## Practical limits

The 0.92-million-parameter starter architecture is intentionally small. Context is 256 tokens by default. Quality depends on subsequent corpus training and evaluation. The existing dataset format is not automatically migrated; current training expects an explicitly chosen prepared-data directory. Optional preparation helpers do not run automatically or overwrite existing destinations.

Training holds prepared records/windows in RAM and snapshots full optimizer state. Inference appends KV tensors rather than using a paged/preallocated cache. These are working correctness-first choices that need profiling before scaling. Cooperative cancellation/disconnect checks occur between generated tokens, not inside tensor kernels.

Checkpoint pointer publication uses same-directory atomic rename; immutable weights use an atomic hard link. Linux local filesystems are the supported deployment target for this mechanism. Host/Origin checks and bounded connections do not replace authentication. The systemd/Git helpers were syntax-checked, not deployed or used to push this rebuild.

The developer seal is attribution. SHA256SUMS covers release files except itself; each generated native checkpoint is hashed by its pointer. JSON checkpoint/run metadata also receives a SHA-256 sidecar. The package changes core code, not the user's data.
