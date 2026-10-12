# Erudite — Roadmap

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

| Stage | Status | Acceptance evidence |
| --- | --- | --- |
| Unified neural/tokenizer/checkpoint core | Implemented | CPU causality, cache, gradient, tokenizer and checkpoint checks |
| Synthetic training-to-serving demonstration | Implemented | Real temporary-input training/evaluation and native generation |
| Existing corpus integration | Deferred | Preserve existing files; explicit input plan and tested read/preparation contract |
| Learned model quality | Unverified | Held-out task results, failure cases and training metadata |
| GPU and scaling | Unverified | Device equivalence, throughput/RAM/VRAM and stability measurements |
| Quantization and GGUF | Not implemented | Matching format/loader integration plus equivalence/quality benchmarks |
| Document retrieval | Not implemented | Retrieval tests, source attribution and context-budget integration |

Do not turn future-stage labels into capability claims. Expand model size/context only after target hardware and appropriate evaluation are known.
