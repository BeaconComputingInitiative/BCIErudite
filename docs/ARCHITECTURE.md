# Erudite — Architecture

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

Erudite separates neural code, training and the local user interface. One native checkpoint contract connects them.

The training path reads explicitly supplied prepared inputs, initializes EruditeLM or resumes its own checkpoint, computes shifted causal-language-model loss, evaluates held-out targets, and writes immutable weights plus atomic best/latest pointers. The checkpoint embeds the exact tokenizer and neural dimensions.

The serving path validates a conversation, admits one generation at a time, loads or refreshes the checkpoint pointer, budgets the actual token context, generates through per-request KV caches, decodes UTF-8 incrementally, and returns NDJSON events. Request metadata is measured; conversation content is not logged.

The Streamlit UI is a thin HTTP client. It does not define another neural network or run training in its event loop. A missing/incompatible checkpoint is an explicit error, not a fabricated assistant reply. Desktop prototypes and the former independent neural paths are retained under legacy/ for reference.

See CORE_RECONSTRUCTION.md for tensor shapes, masks, loss, caching, persistence and IFF contracts; API_SPEC.md for endpoint behavior. Existing datasets are preserved and require a separate explicit integration decision.
