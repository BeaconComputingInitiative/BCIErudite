# Erudite — Core Reconstruction

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

## Neural topology

The active neural network is `EruditeLM`. It initializes every learned embedding/projection with zero-mean Gaussian values (standard deviation 0.02); RMSNorm scales start at one. There is no inherited base model. One tokenizer determines the model vocabulary and all token meanings.

| Stage | Tensor/output | Implemented behavior |
| --- | --- | --- |
| Input | int64 `[batch, time]` | Byte/BPE IDs plus reserved BOS/EOS/padding/system/user/assistant IDs |
| Token embedding | `[batch, time, width]` | Learned embedding table |
| Attention branch | Same shape | Pre-RMSNorm → fused QKV → rotary query/key positions → causal SDPA → projection → residual addition |
| Feed-forward branch | Same shape | Pre-RMSNorm → SwiGLU gates with hidden size rounded to a multiple of eight → down projection → residual addition |
| Repeated blocks | Same shape | Four blocks in the starter config; architecture is stored in every checkpoint |
| Output | `[batch, time, vocabulary]` | RMSNorm and projection sharing the exact embedding weight parameter |
| Loss | Scalar | Shifted next-token cross entropy, ignoring `-100` padding/masked labels |

Attention heads must divide width exactly and each head dimension must be even. The model context is an enforced upper bound shared by rotary buffers, cached attention and generation. No positional slicing silently exceeds it. Batch-one generation is explicit; batched sampling is not silently treated as one scalar.

The initial config is width 128/layers 4/heads 4/context 256/vocabulary 1,024. This has 922,752 unique learned parameters. A larger network is a deliberate configuration change with fresh compatible checkpoints, not a filename tweak.

## Causality and inference reconstruction

Without cache, SDPA applies a causal mask. With a cached prefix and one new token, all prefix/current keys are visible. With several new tokens, row i may access a key iff its position is at most cached_prefix_length+i. The cached/full equivalence tests exercise both paths.

Each request owns its KV-cache list; caches are not stored in a shared neural-model object. The local API admits only one active generation. Cache appends currently concatenate tensors rather than using an optimized preallocated paged cache; performance at long context still needs profiling. RoPE and RMSNorm/SwiGLU replace the old independently defined model paths.

Inference uses the system prompt recorded by the dataset/training checkpoint. Old whole conversation pairs are dropped iff exact token budgeting requires it; if system/current user/output reserve cannot fit, inference fails explicitly. Generation forbids structural markers inside the answer, stops on EOS or token cap, and decodes UTF-8 incrementally so partial byte tokens do not become spurious replacement characters.

## Training objective and data reconstruction

Raw records require text, valid role/content messages, or explicit instruction/response fields. Telemetry dictionaries are not serialized into fake answer examples. Preparation rejects bad rows with a source filename/line number. It removes duplicates by canonical training payload, not by IDs or other metadata. Byte-identical files are skipped after their source hash is recorded.

Exact user-prompt groups remain in a single partition even when their answers differ. Partitions are assigned before tokenization/windowing. The tokenizer learns only from training payloads, subject to a recorded character budget. Semantic/source-family overlap remains possible and is explicitly not certified away by exact hashing.

Default objective `all` trains all shifted target tokens. Optional objective `assistant` excludes prose records and masks system/user targets, using the same already prepared tokenizer and token sequences. Assistant-mask coverage is verified by tests. Padding follows real tokens and is ignored in loss; causal attention prevents real positions from attending to that trailing padding.

Training uses AdamW, warmup/cosine decay, finite-loss/gradient checks and clipping. Full validation cross entropy is weighted by actual unmasked target counts. The best validation snapshot is distinct from the latest snapshot; test-set loss is not used for model selection. Full corpus training windows are kept in RAM in this first reconstruction; streaming/sharded data loading is a later optimization.

## Checkpoint reconstruction

A checkpoint embeds architecture, learned weights, complete tokenizer specification/fingerprint, prepared-data fingerprint, system prompt, training metadata, optimizer state and RNG state. Loading uses `weights_only=True`, verifies bytes before loading, reconstructs the exact architecture and then moves the model to the requested device.

Weights use immutable `step-XXXXXXXX.pt` names. Publication uses a temporary file, flush/fsync, and an atomic no-overwrite hard link inside the same run directory. Atomic JSON pointers (`latest.json`, `best.json`) become current after the referenced file is complete. The pointer publication requires a filesystem supporting same-directory atomic replacement and hard links; Linux local filesystems are the intended deployment target. Parent-directory crash durability has not been power-failure tested.

Tokenizers, architectures or datasets that differ during resume are rejected. Resuming targets a fresh run directory. Native continuation restores optimizer/RNG state; GPU determinism and bitwise equality across versions/devices are not claimed. Changing total steps or objective changes the training procedure deliberately.

Old raw state dictionaries and old BPE artifacts are preserved under legacy/ but do not satisfy the new contract. Automatic old-weight migration is not provided: the former architectures and token meanings differ.

## Meaning of IFF contracts

IFF comments accompany concrete predicates rather than being used as a substitute for checks. Architecture is admitted iff dimensions/context are valid; records iff schema is valid; prepared data iff hashes/tokenizer/partition contracts agree; checkpoints iff format/path/hash/tokenizer/architecture agree; generation iff its context and sampling settings fit. These are engineering contracts, not a security classifier or proof of language-model competence.
