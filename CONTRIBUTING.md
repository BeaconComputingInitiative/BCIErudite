# Contributing to Erudite

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

Keep model/tokenizer/checkpoint compatibility explicit. Trace failures through inputs, tokenization, neural execution, loss, persistence and inference before patching. Separate root causes from secondary symptoms. Include relevant regression tests for changed contracts and report actual verification output.

Run `python -m unittest discover -s tests -v` for relevant core changes. Synthetic checks must use temporary directories and must not rewrite repository datasets or overwrite prior run outputs. Update architecture/API/transparency documentation when behavior changes. Keep private data, credentials and generated weights out of commits.

After intentional source edits, regenerate SHA256SUMS with `python scripts/update_hashes.py`, then run `python scripts/verify_hashes.py`. Checkpoint hashes establish byte integrity relative to a trusted pointer, not authorship or security certification.

Use a branch and reviewable pull request. Describe concrete behavior and test results; do not claim trained-model competence from a toy demonstration. The current ownership notice reserves rights and does not grant general redistribution permission.
