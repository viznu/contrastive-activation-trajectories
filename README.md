# Contrastive Activation Trajectories

A pair-aware contrastive learning approach for monitoring alignment-relevant
behavioral state in transformer internal activations.

## Thesis

We train a small transformer encoder over a target LLM's **full layer-wise
residual-stream trajectory** — one activation vector per transformer block,
sampled at a chosen token position — with a SupCon-style InfoNCE contrastive
objective over behavior-paired prompts.

The resulting behavioral-state latent:

- matches cross-validation-selected single-layer activation probes on AUROC
  **without requiring layer selection**;
- is roughly **32× more label-efficient** than raw-activation probes at
  downstream probe time;
- **transfers zero-shot** across content distribution, behavioral axis, and
  content type.

## Headline numbers (Qwen2.5-3B-Instruct)

### Main comparison on Instructed-Pairs

Probe fit on 488 fact-grouped train trajectories; test is 124 held-out
paired-examples from 62 unseen facts.

| Probe | AUROC | recall@1%FPR |
|---|---|---|
| Single-layer (raw target layer 24)       | 0.9917 | 0.9677 |
| All-layers-concat (flattened `L·d`)      | 0.9977 | 0.9677 |
| Supervised transformer-over-layers       | 0.9922 | 0.8548 |
| **Contrastive trajectory encoder** (ours) | **0.9948** | **0.9839** |

![main](results/figures/main_comparison.png)

The contrastive encoder matches the all-layers-concat baseline and beats
single-layer and the supervised transformer on recall at 1% FPR — without
a per-task layer sweep.

### Label efficiency

5 seeds per cell, fixed held-out test set of 124 paired examples.

| N_train | single_layer | all_layers_concat | transformer | **contrastive_encoder** |
|---|---|---|---|---|
| 5   | 0.75 ± 0.06 | 0.59 ± 0.03 | 0.55 ± 0.02 | **1.00 ± 0.00** |
| 10  | 0.73 ± 0.04 | 0.60 ± 0.01 | 0.56 ± 0.02 | 0.99 ± 0.00 |
| 20  | 0.88 ± 0.03 | 0.72 ± 0.07 | 0.62 ± 0.02 | 1.00 ± 0.00 |
| 40  | 0.92 ± 0.03 | 0.77 ± 0.06 | 0.67 ± 0.05 | 0.99 ± 0.00 |
| 80  | 0.97 ± 0.01 | 0.93 ± 0.05 | 0.81 ± 0.03 | 1.00 ± 0.00 |
| 160 | 0.99 ± 0.00 | 0.98 ± 0.01 | 0.90 ± 0.02 | 0.99 ± 0.00 |
| 320 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.97 ± 0.01 | 0.99 ± 0.00 |

![few-shot](results/figures/few_shot.png)

The contrastive encoder hits **1.00 AUROC from 5 labels**. The single-layer
baseline needs 160 labels to match; all-layers-concat needs 320. The
supervised transformer-over-layers doesn't catch up within the sweep.
That's roughly a 32× label-efficiency advantage.

### Zero-shot transfer

Probe is fit on IP-true train (honest/deceptive framing of 244 true facts).
The **same probe is then applied, without refit**, to three target settings.

| Target | AUROC | recall@1%FPR | Gap from in-dist |
|---|---|---|---|
| IP-false (content distribution)                 | 0.9900 | 0.8170 | 0.005 |
| Refuse-vs-comply on AdvBench (distinct prefills) | 0.9912 | 0.9500 | 0.004 |
| Refuse-vs-comply with lexically-matched prefills | 0.9479 | 0.8400 | 0.047 |

![transfer](results/figures/transfer.png)

The learned behavioral latent transfers across:
1. **Content distribution** — different facts (true → false).
2. **Behavioral axis** — from honest-vs-deceptive framing to
   refuse-vs-comply on harmful requests.
3. **Content type** — factual statements → harmful prompts.
4. **Lexical robustness** — holds when refuse and comply prefills begin
   with an identical 7-token prefix before diverging, so the probe is
   not relying on "I'm sorry" / "Sure" lexical markers.

## Repository layout

```
src/
  extract/
    extract_activations.py       generic last-prompt-token extraction
    extract_instructed_pairs.py  Zou-RepEng honest/deceptive pairs
    extract_refusal_pairs.py     refuse/comply pairs on AdvBench
    inspect_activations.py       shape / norm sanity checks
  encoder/
    model.py                     TrajectoryEncoder (transformer over L-slot)
    train.py                     SupCon-style InfoNCE training loop
  probes/
    probes.py                    SingleLayer / AllLayersConcat /
                                 TransformerOverLayers / ContrastiveEncoder
    evaluate.py                  fit all probes on one split, compare
    sweep_few_shot.py            AUROC-vs-N_train with seed variance
    transfer_test.py             zero-shot transfer across datasets
  make_figures.py                paper-ready plots from result JSONs
data/external/
  facts_true_false.csv           Zou et al. 2023 facts dataset
results/
  *.json                         paper-ready metric files
  figures/                       generated plots
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Extract per-layer residual-stream trajectories from the target LLM.
python -m src.extract.extract_instructed_pairs \
  --n_facts 306 --output data/instructed_pairs.pt
python -m src.extract.extract_instructed_pairs \
  --n_facts 306 --fact_label 0 --output data/instructed_pairs_false.pt
python -m src.extract.extract_refusal_pairs \
  --n_prompts 100 --prefill_mode distinct --output data/refusal_pairs.pt
python -m src.extract.extract_refusal_pairs \
  --n_prompts 100 --prefill_mode matched --output data/refusal_pairs_matched.pt

# 2. Train the contrastive trajectory encoder.
python -m src.encoder.train \
  --data data/instructed_pairs.pt --output results/encoder_ip_infonce.pt \
  --epochs 40 --temperature 0.07 --pool layer --layer_idx 24

# 3. Evaluate against the three raw-activation baselines.
python -m src.probes.evaluate \
  --data data/instructed_pairs.pt --output results/eval_ip_layer24.json \
  --group_field fact_ids --encoder_ckpt results/encoder_ip_infonce.pt \
  --encoder_pool layer --encoder_layer_idx 24

# 4. Label-efficiency sweep.
python -m src.probes.sweep_few_shot \
  --data data/instructed_pairs.pt --output results/few_shot_ip.json \
  --group_field fact_ids --encoder_ckpt results/encoder_ip_infonce.pt \
  --encoder_pool layer --encoder_layer_idx 24

# 5. Zero-shot transfer.
python -m src.probes.transfer_test \
  --encoder_ckpt results/encoder_ip_infonce.pt \
  --src data/instructed_pairs.pt --tgt data/instructed_pairs_false.pt \
  --output results/transfer_content_layer24.json
python -m src.probes.transfer_test \
  --encoder_ckpt results/encoder_ip_infonce.pt \
  --src data/instructed_pairs.pt --tgt data/refusal_pairs.pt \
  --output results/transfer_task_refusal_distinct_layer24.json
python -m src.probes.transfer_test \
  --encoder_ckpt results/encoder_ip_infonce.pt \
  --src data/instructed_pairs.pt --tgt data/refusal_pairs_matched.pt \
  --output results/transfer_task_refusal_matched_layer24.json

# 6. Regenerate figures.
python -m src.make_figures
```

## Target model and hardware

All results are on `Qwen/Qwen2.5-3B-Instruct` (36 transformer blocks, hidden
size 2048) on a Mac Studio with MPS backend. Forward passes use fp16; saved
trajectories and the contrastive encoder use fp32.

## License

MIT.
