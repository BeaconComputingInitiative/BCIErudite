# Erudite — Migration

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

The active project uses core/inference/EruditeLM and one embedded native BPE contract. The old independent root Transformer, character-code tokenizer path and desktop workers are retained under legacy/. Old code remains available for inspection but is excluded from active entry points and test discovery.

Root app.py delegates to ui/app.py. Root train_pipeline.py delegates to training/train_model.py. Model loading now takes a native pointer JSON (best.json/latest.json), not an old bare tensor state dictionary or standalone old tokenizer JSON.

The old architectures and token meanings differ from the reconstructed model. Old checkpoints are not silently converted or reused. Start with a fresh native training run, or design/test a specific migration separately. No dataset migration is applied by this pull request: dataset/ blobs are preserved exactly.

Extracting a source archive into a separate folder is the safest manual trial. GitHub reviewers can check out the reconstruction branch and run the synthetic demo without using dataset/. Merge remains a separate review decision.
