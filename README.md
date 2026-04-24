# Contrastive Activation Trajectories

A pair-aware contrastive learning approach for monitoring alignment-relevant
behavioral state in transformer internal activations.

## Thesis

Train a small transformer encoder over a target LLM's **full layer-wise
residual-stream trajectory** with a pair-aware contrastive (SupCon-style
InfoNCE) objective. The learned latent exposes an aligned-vs-misaligned
behavioral axis that:

- matches CV-selected single-layer activation probes on AUROC without
  requiring layer selection;
- is dramatically more label-efficient than direct activation probes;
- transfers zero-shot across content distribution, behavioral axis, and
  content type.

## Structure

```
src/
  extract/    forward-hook extraction of per-layer activations at
              chosen token positions for paired prompts.
  encoder/    trajectory encoder + contrastive training loop.
  probes/     baseline activation probes (single-layer, all-layers-
              concat, supervised transformer-over-layers) and the
              contrastive encoder probe; evaluation, few-shot sweep,
              transfer test.
data/external/   small reference datasets (Zou et al. facts CSV).
results/         experiment result JSONs and figures.
```

## Pipeline

1. **Extract paired activation trajectories** from a target LLM on a
   paired-prompt task (e.g. honest/deceptive framing of factual claims,
   refuse/comply responses to harmful requests).
2. **Train a trajectory encoder** with InfoNCE over paired trajectories.
3. **Evaluate** with a small linear probe on the encoder output; compare
   against raw-activation baselines; test label efficiency and transfer.
