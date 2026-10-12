# Erudite — Training and Evaluation

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

## Demonstration versus corpus training

`scripts/smoke_run.py` performs real CPU training on temporary synthetic sentences. It verifies the engineering chain without accessing existing datasets. Its output is not a useful pretrained assistant. The main training configuration expects an explicitly selected prepared directory; it never automatically transforms existing dataset files.

## Objectives

`objective: all` learns every shifted next-token target in text/chat windows. `objective: assistant` uses chat windows and ignores system/user targets. Padding targets use -100 and do not contribute to cross entropy. Model architecture and tokenizer do not change when the objective changes.

Training uses AdamW, finite-loss/gradient checks, clipping, warmup/cosine decay, and token-weighted validation cross entropy. The best validation checkpoint is separate from the latest checkpoint. Test-set evaluation is not used for checkpoint selection.

## Native continuation

Copy config/training_config.yaml into a new configuration, preserve prepared-data/tokenizer/architecture identity, set a fresh output_dir and a larger total steps count, then run:

```bash
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
python -m training.fine_tune --checkpoint models/runs/first/latest.json \
  --config config/continuation.yaml
```

Continuation restores Erudite's own weights, optimizer and RNG states; it does not use a pretrained external base, PEFT or LoRA. Changing total steps changes the learning-rate schedule; bitwise equivalence across devices/library versions is not claimed.

## Optional preparation interface

Preparation accepts text, validated message conversations or explicit instruction/response objects. It rejects malformed rows, removes exact payload duplicates, keeps repeated prompt groups together, splits before tokenization/windowing, and trains BPE only on training inputs. This optional code does not run automatically. Output must be a new directory, preserving previous artifacts and original files.

Exact hashes do not detect semantic/source-family overlap. Before real training, independently assess corpus ownership, answer quality, split validity, representativeness, and useful evaluation tasks. Full prepared windows currently reside in RAM; larger-scale streaming/sharding needs profiling and an additional implementation step.
