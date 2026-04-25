# Label-Efficient Alignment Monitoring via Behavior-Paired Contrastive Pretraining

A controlled study of contrastive representation learning over a target LLM's
internal activations, for monitoring alignment-relevant behavioral state.

## Findings

We pretrain a simple encoder on behavior-paired activations from a target
LLM (e.g. honest vs. deceptive framing of factual claims) using a SupCon /
InfoNCE objective. A small linear probe on the encoded features then:

1. **Matches CV-selected single-layer activation probes on AUROC** without
   requiring layer selection.
2. **Reaches 0.99+ AUROC from as few as 5 downstream labels** — roughly
   32× more label-efficient than fitting a logistic probe directly on raw
   activations from scratch.
3. **Transfers zero-shot** across content distribution, behavioral axis,
   and content type.
4. **Is architecture-insensitive at this task and scale.** A single
   Linear(2048, 256) projection trained with SupCon achieves the same
   label-efficiency and transfer as an MLP on the flattened trajectory
   or a transformer over layer slots. The contrastive objective and pair
   structure are doing the work; the encoder architecture is cosmetic.

What we are *not* claiming:
- We are not claiming a new architecture; the architecture ablation shows
  none of the variants we tested (linear, MLP, transformer-over-layers)
  is required.
- We are not claiming a general deception or alignment detector; the
  results are on controlled paired-prompt benchmarks.

## Headline numbers (Qwen2.5-3B-Instruct)

### Main probe comparison on Instructed-Pairs (IP)

Probe fit on 488 fact-grouped train trajectories; test is 124 held-out
paired-examples from 62 unseen facts.

| Probe | AUROC | recall@1%FPR |
|---|---|---|
| Single-layer (raw target layer 24, no pretraining) | 0.9917 | 0.9677 |
| All-layers-concat (raw flattened, no pretraining)  | 0.9977 | 0.9677 |
| Supervised transformer-over-layers (no pretraining) | 0.9922 | 0.8548 |
| **Contrastive-pretrained encoder** (any architecture, see below) | 0.99+ | 0.93–0.97 |

![main](results/figures/main_comparison.png)

### Architecture ablation — contrastive objective held fixed

All four trained on the same IP-true train split with identical SupCon /
InfoNCE hyperparameters (40 epochs, T=0.07, AdamW lr=1e-3, wd=0.05,
d_model=256). Only the encoder architecture differs.

| Encoder | Params | AUROC | recall@1%FPR |
|---|---|---|---|
| Linear (layer 24 only)        | 0.5M | 0.9940 | 0.9677 |
| Linear (flattened trajectory) | 19M  | 0.9984 | 0.9677 |
| MLP (flattened trajectory)    | 19M  | 0.9971 | 0.9355 |
| Transformer over layer slots  | 3.7M | 0.9896 | 0.9355 |

![ablation](results/figures/architecture_ablation.png)

All four are within ~0.01 AUROC of each other. **The simplest variant — a
single Linear projection on one layer — matches the most complex.** The
load-bearing factor is the contrastive objective + pair structure, not
the encoder.

### Label efficiency

5 seeds per cell, fixed held-out test set of 124 paired examples.
Contrastive-pretrained encoder uses our 4-layer transformer, but per the
ablation any of the four architectures gives essentially the same curve.

| N_train | single_layer (raw) | all_layers_concat (raw) | transformer (raw) | **contrastive_pretrained** |
|---|---|---|---|---|
| 5   | 0.75 ± 0.06 | 0.59 ± 0.03 | 0.55 ± 0.02 | **1.00 ± 0.00** |
| 10  | 0.73 ± 0.04 | 0.60 ± 0.01 | 0.56 ± 0.02 | 0.99 ± 0.00 |
| 20  | 0.88 ± 0.03 | 0.72 ± 0.07 | 0.62 ± 0.02 | 1.00 ± 0.00 |
| 40  | 0.92 ± 0.03 | 0.77 ± 0.06 | 0.67 ± 0.05 | 0.99 ± 0.00 |
| 80  | 0.97 ± 0.01 | 0.93 ± 0.05 | 0.81 ± 0.03 | 1.00 ± 0.00 |
| 160 | 0.99 ± 0.00 | 0.98 ± 0.01 | 0.90 ± 0.02 | 0.99 ± 0.00 |
| 320 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.97 ± 0.01 | 0.99 ± 0.00 |

![few-shot](results/figures/few_shot.png)

The contrastive-pretrained encoder hits **1.00 AUROC from 5 labels**
downstream. Raw-activation probes need many more (single-layer needs ~160,
all-layers-concat needs ~320). Note this is downstream label efficiency
*after* pretraining the encoder on 488 paired labels — not a fewer-total-
labels claim. The pretrained encoder is reusable across downstream
evaluations.

### Zero-shot transfer

Probe fit on IP-true-train (244 facts × honest/deceptive). The **same
probe** is then applied, without refit, to three target settings.

| Target | AUROC | recall@1%FPR | Gap from in-dist |
|---|---|---|---|
| IP-false (different content distribution)         | 0.9900 | 0.8170 | 0.005 |
| Refuse-vs-comply on AdvBench (distinct prefills)  | 0.9912 | 0.9500 | 0.004 |
| Refuse-vs-comply with lexically-matched prefills  | 0.9479 | 0.8400 | 0.047 |

![transfer](results/figures/transfer.png)

The behavioral-state direction transfers across:
- **Content distribution** — honest/deceptive on different facts (true → false).
- **Behavioral axis** — from honest-vs-deceptive framing to
  refuse-vs-comply on harmful requests.
- **Content type** — factual claims → harmful prompts.
- **Lexical robustness** — holds when refuse and comply prefills begin
  with an identical 7-token prefix before diverging.

## Repository layout

```
src/
  extract/
    extract_activations.py          generic last-prompt-token extraction
    extract_instructed_pairs.py     Zou-RepEng honest/deceptive pairs
    extract_refusal_pairs.py        refuse/comply pairs on AdvBench
    inspect_activations.py          shape / norm sanity checks
  encoder/
    model.py                        TrajectoryEncoder (transformer over L-slot)
    baselines.py                    LinearSingleLayer / LinearConcat / MLPConcat
    train.py                        SupCon-style InfoNCE training loop
  probes/
    probes.py                       SingleLayer / AllLayersConcat /
                                    TransformerOverLayers / ContrastiveEncoder
    evaluate.py                     fit all probes on one split, compare
    sweep_few_shot.py               AUROC vs N_train, with seed variance
    transfer_test.py                zero-shot transfer across datasets
  experiments/
    fair_contrastive_baselines.py   architecture ablation under fixed objective
  make_figures.py                   plots generated from result JSONs
data/external/
  facts_true_false.csv              Zou et al. 2023 facts dataset
results/
  *.json                            metric files written by the scripts above
  figures/                          generated plots
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

# 3. Architecture ablation (the headline finding for this paper).
python -m src.experiments.fair_contrastive_baselines \
  --output results/fair_contrastive_baselines.json

# 4. Evaluate against the three raw-activation baselines.
python -m src.probes.evaluate \
  --data data/instructed_pairs.pt --output results/eval_ip_layer24.json \
  --group_field fact_ids --encoder_ckpt results/encoder_ip_infonce.pt \
  --encoder_pool layer --encoder_layer_idx 24

# 5. Label-efficiency sweep.
python -m src.probes.sweep_few_shot \
  --data data/instructed_pairs.pt --output results/few_shot_ip.json \
  --group_field fact_ids --encoder_ckpt results/encoder_ip_infonce.pt \
  --encoder_pool layer --encoder_layer_idx 24

# 6. Zero-shot transfer.
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

# 7. Regenerate figures.
python -m src.make_figures
```

## Target model and hardware

All results are on `Qwen/Qwen2.5-3B-Instruct` (36 transformer blocks, hidden
size 2048) on a Mac Studio with MPS backend. Forward passes use fp16; saved
trajectories and the contrastive encoder use fp32.

## Limitations

- Single target model. Cross-model generalization untested.
- Two paired-prompt task families (Instructed-Pairs honest/deceptive,
  AdvBench refuse/comply). Harder behavioral contrasts (sycophancy,
  sandbagging, deeper deception) untested.
- Contrastive pretraining uses paired supervision; this is not a fully
  unsupervised method.

## License

MIT.
