# Paper-code audit

This file records the differences found between the final (camera-ready) paper,
the released code, and the archived experimental workspace the code was built
from, and how each was handled.

Rules applied throughout:

* No experiment was re-run and no result was recomputed or altered. Published
  numbers are copied from the paper.
* The code was changed only where the final paper or the archived final
  experiments demonstrably required it; every such change is listed below.
* Where a discrepancy could not be resolved from the available evidence, the
  archived behaviour was preserved and the discrepancy is recorded here.

Severity: **high** means a re-run behaves differently from the runs behind the
published numbers; **medium** means a reader could be misled; **low** is
organisational.

---

## 1. N4 bias-field correction

**Severity: medium.** The final paper's preprocessing does not mention N4, and
the archived preprocessing notebook has no N4 step. An earlier release draft
enabled N4 by default.

**Handling.** The N4 implementation is kept as an optional step, but
`apply_n4` now defaults to `false` (configuration file, `scripts/preprocess.py`
fallback and `PreprocessConfig`). The default pipeline matches the paper and
the archived runs.

## 2. In-plane resampling, intensity clipping

**Severity: none (checked, consistent).** The final paper states RAS
reorientation, 0.8 x 0.8 mm in-plane resampling (B-spline image, nearest
neighbour mask), a 5th-percentile foreground threshold, clipping to the
0.5th-99.5th foreground percentiles, Z-score normalisation and clipping to
[-5, 5]. The code and the archived notebook implement exactly this. The
architecture figure is Figure 1 of the final paper (`docs/architecture.png`).

## 3. Tumour-centred crop

**Severity: low.** The crop is centred on the ground-truth tumour bounding box
(+40 px padding), with an image-centre fallback for tumour-free slices, as the
final paper describes. The mask is used only during preprocessing, never as a
model input; a deployment would need a tumour localisation step.

## 4. Classification-loss slices

See item 7.

## 5. Model input size: 256 and 224 in the paper, one size in the code

**Severity: high.** The final paper states 256 x 256 inputs for the CNN pathway
and 224 x 224 for the Swin pathway. The archived workspace preprocesses two
variants (`CNN_SIZE = 256`, `SWIN_SIZE = 224`). Its centralised and federated
Swin Hybrid notebooks both load the 224 x 224 ("swin") variant, and the model
code feeds a single tensor to both encoder branches; there is no dual-resolution
input. The archived Swin-Tiny model could only run at 224 x 224.

**Handling.** The released model takes `img_size` and passes it to `timm`, so
224 and 256 both build. The default is now **224**, the resolution of the
Swin Hybrid runs (configs, script fallbacks, model and preprocessing
defaults, diagram, documentation). The 256 variant belonged to the CNN-only
baselines, which are not in this repository. Training at 256 remains possible
via `input_size`, but is not what produced the reported Swin Hybrid numbers.
The paper's wording ("256 for the CNN pathway") is not literally implemented;
this is left as a documented difference.

## 6. ResNet-34 learning-rate multiplier was configured but not applied

**Severity: high.** Section 4.2 reports learning-rate multipliers of 0.15 for
the ResNet-34 branch and 0.30 for the Swin-Tiny branch. In the archived
optimiser builder every parameter whose name lacks `swin` falls into the
base-learning-rate group, so the ResNet-34 branch trained at 2e-5 and
`enc_lr_ratio: 0.15` was never read. Only the Swin multiplier (0.30) took
effect, in both the centralised and the federated runs.

**Handling.** The recorded runs produced the reported numbers, so the release
reproduces them: `enc_lr_ratio` is 1.0 in `configs/swin_hybrid.yaml` and
`configs/federated.yaml` (and as the code fallback), which puts the ResNet-34
branch at the base rate exactly as the archived builder did. The paper's 0.15
is not what ran. Setting `enc_lr_ratio: 0.15` applies it, but that is a
different experiment from the one behind the tables.

## 7. Classification loss is supervised on one slice per patient

**Severity: medium.** Section 3.3 states that the classification objective "is
evaluated only for tumor-bearing slices". The archived dataset narrows this
further: `use_cls_loss` is set from `is_best_slice`, so only the slice with the
largest tumour area per patient contributes to the classification loss, even
though the preprocessing step records the broader `has_tumor or is_best` flag.

**Handling.** The implementation is preserved, because it is what produced the
results, and the narrower rule is stated explicitly in the dataset docstring
and in the code comment. The published description is the broader of the two.

## 8. Classification-loss ramp

**Severity: medium for federated training.** Section 3.3 says the classification
weight "is gradually increased during the first 20 cumulative training
epochs". The configured `cls_rampup_epochs: 10` produces a weight of zero for
ten epochs, a linear rise over the next ten, and the full weight from epoch 20.
This agrees for centralised training; the relationship is documented in
`classification_weight`.

**Federated classification-loss ramp.** The final paper states (Sections 3.3 and 4.2) that the
classification loss weight is gradually introduced during the first 20
cumulative training epochs. The archived federated client uses a fixed
classification loss weight of 0.20 from the start, and the released federated
client preserves that behavior rather than changing the implementation
associated with the reported federated results. The paper's statement is
therefore accurate for centralised training only. The centralized trainer
implements the documented 20-epoch classification-loss ramp.

## 9. Result tables: provenance and inconsistencies

**Severity: medium.** The values in `results/` and the README are copied from
Tables 2 and 3 of the final paper. They were compared with the recorded
artifacts:

* **Centralised rows.** All seven centralised rows of Table 3 and the Swin
  Hybrid row of Table 2 (0.8088 / 0.9286 / 0.8627) match
  `Centralized_Model` `final_model(ablation_study)/figures/
  centralized_vs_federated_ablation.csv`, and the centralised values
  hard-coded in the archived `09_fl_analysis` notebook. An earlier release
  draft carried 0.8076 / 0.8214 / 0.8138 for the Swin Hybrid / None row. That
  figure appears only in the `Publishable_model` copy of the same CSV, which
  contradicts that repository's own notebook; it is not used. Corrected to the
  paper value.
* **Federated rows, matching.** None, GIN Only, Geo+Bias and Geo+Noise match
  `centralized_vs_federated_ablation.csv`. Geo+Elastic (0.8100 / 0.8931 /
  0.8474) and Geo+Gamma (0.7987 / 0.8575) match the validation columns of
  `Publishable_model` `figures/fl/fl_summary.csv`.
* **Combined-score rounding.** Recomputing 0.55 DSC + 0.45 AUC from the
  reported DSC and AUC gives 0.8225 for the federated Augmentation Stack and
  0.8252 for federated Geo+Gamma. The earlier draft had 0.8220 and 0.8250;
  both were corrected to the paper and the arithmetic. All other rows
  recompute to the reported values.
* **UNRESOLVED: federated None and Augmentation Stack.** The paper reports
  0.7908 / 0.7716 / 0.7822 and 0.8220 / 0.8230. `fl_summary.csv` (the final
  federated run) gives 0.7974 / 0.7691 / 0.7846 and 0.8293 / 0.7893 / 0.8113
  for these. The paper values appear only in `centralized_vs_federated_
  ablation.csv`. The paper's federated table therefore mixes two sources. The
  paper values were kept; the authors should confirm which run each row
  comes from.
* **UNRESOLVED: conflicting CSV rows.** The same CSV gives federated
  Geo+Elastic 0.8067 / 0.9476 / 0.8701 and Geo+Gamma 0.8175 / 0.8951 / 0.8524,
  and the `Publishable_model` README quotes 0.870 / 0.948. These disagree with
  `fl_summary.csv` and the paper and were not used.
* **UNRESOLVED: split used.** The paper does not say whether the table values
  are validation or test metrics. The federated Geo+Elastic and Geo+Gamma values
  equal the validation columns of `fl_summary.csv`; the test columns of that
  file differ (e.g. Geo+Elastic test 0.7938 / 0.9012). Held-out test results
  in `test_results.csv` files also differ from the paper tables.
* **Federated coverage.** `fl_summary.csv` covers four of the seven
  strategies (see item 11); the other three federated rows rest on the CSV only.
* **Paper-internal inconsistencies (paper not modified).** Section 5.2 says
  Geo+Elastic has the highest centralised AUC (0.9196), but Table 3 lists None
  at 0.9286. The Introduction says six augmentation strategies are evaluated;
  Sections 4.4 and 5.2 and Table 3 list seven. The Discussion sets the
  centralised-DSC leader (0.8487) beside the federated combined score
  (0.8474) in one sentence. The authors may wish to correct these.

## 10. Other result files in the workspace

**Severity: medium.** `ablation_table.csv`, `test_results.csv` and the other
files listed above come from other runs or splits and disagree with the paper.
None were carried into the release; `results/` contains paper values only.

## 11. Federated runs cover four of the seven augmentation strategies

**Severity: low.** Table 3 reports seven federated configurations. The archived
federated notebooks expose four (`no_aug`, `full_aug`, `geo_elastic`,
`geo_gamma`); the remaining three (`gin_only`, `geo_bias`, `geo_noise`) exist
as augmentation pipelines but have no federated runner.

**Handling.** The released federated path takes the strategy by name from the
configuration or `--augmentation`, so all seven run through the same code. This
generalises the runner; it does not add or alter any result.

## 12. GIN-only pipeline uses the ResNet-era GIN variant

**Severity: medium for the "GIN Only" row.** The archived Swin "GIN Only"
experiments (`gin_only=True`) built `GINOnlyPipeline`, which instantiates
`ResNetGINAugmentation(n_layers=3, p=0.30)`: 2-5 channel layers, weight
initialisation std 0.5, blend factor 0.08-0.22. The Augmentation Stack uses
the stronger `GINAugmentation(n_layers=4, p=0.40)`.

**Handling.** The release restores the archived behaviour: `ResNetGINAugmentation`
is included and `GINOnlyPipeline` uses it, with no geometric companion
transform. An earlier release draft had substituted the stronger variant
(p=0.70), which does not match the recorded "GIN Only" runs.

## 13. Hyperparameters present in the code but absent from the paper

**Severity: low.** Several values are needed to run the code and are not stated
in the paper: Focal Tversky `alpha=0.40, beta=0.60, gamma=0.75`; the segmentation
loss weights `lambda_dice=0.70, lambda_ce=0.30, tversky_weight=0.20,
boundary_weight=0.03`; deep-supervision weights `(0.50, 0.25)`; focal
`gamma=2.0` with `mibc_alpha=1.6` and label smoothing 0.05; MixStyle `p=0.35,
alpha=0.1`; the tumour-slice oversampling weights; and the intensity clip of
+/- 5 after Z-scoring.

**Handling.** All are carried over unchanged from the archived configuration,
surfaced in `configs/` where they are tunable, and documented in the
docstrings. They are the values used for the reported experiments.

---

## Verified as consistent

Checked against the paper and found to agree, so no action was needed:

* five-channel 2.5D input, two adjacent slices above and below the target;
* input adapter projecting 5 channels to 3, with mean initialisation;
* ImageNet-pretrained ResNet-34 and Swin-Tiny dual-stream encoder;
* MixStyle applied to the CNN branch only;
* intermediate feature fusion by concatenation and 1x1 projection;
* attention-gated decoder with deep supervision at two depths;
* segmentation-guided attention pooling with the segmentation map detached
  before pooling;
* joint objective `L = 0.80 L_seg + 0.20 L_cls`;
* segmentation loss combining Dice, BCE, Focal Tversky and boundary terms with
  auxiliary deep-supervision terms, and focal cross-entropy for classification;
* Geo+Elastic parameters: flip p=0.5, rotation +/-15 degrees p=0.70, elastic
  sigma=5.0, alpha=18.0, p=0.20;
* AdamW, warmup-cosine schedule, gradient clipping at max norm 1.0, base
  learning rate 2e-5, Swin multiplier 0.30, weight decay 1e-4, effective batch
  size 24 (batch 6 x 4 accumulation steps);
* federated protocol: 4 clients, full participation each round, 3 local epochs,
  50 rounds, sample-size-weighted FedAvg;
* model selection on the highest validation combined score;
* DSC computed on non-empty ground-truth slices, AUC at the patient level by
  averaging slice probabilities, `S_comb = 0.55 DSC + 0.45 AUC`;
* FP32 training for the Swin Hybrid.

---

## Issues found by running the released code

These were found while validating this repository, not in the paper. They are
fixed here and recorded so the authors know the released code differs from the
archive in these respects.

* **Swin input-size assertion.** See item 5: the archived model cannot run at
  256 x 256. The released model threads `img_size` through to `timm`.
* **Data split.** An earlier release draft allocated patients with its own
  rounding rules, which does not reproduce the recorded split. `scripts/
  make_splits.py` now implements the archived procedure (seed 42, strata by
  centre and class, patients visited in ID order, `round(0.20 n)` test and
  `round(0.10 n)` validation patients, at least one each). Applied to the 221
  archived patient IDs it reproduces all 221 archived assignments
  (157 / 22 / 42 patients).
* **Background-slice sampling.** The archived preprocessing draws the two
  tumour-free slices per patient from one seeded generator shared by all
  patients; the release script now does the same. The draw depends on the order
  in which patients are processed, so identical slices require the same
  dataset listing order.
* **Script imports.** The archived pipeline ran from notebooks that mutated
  `sys.path` at the top of each one. Each released script instead resolves the
  repository root from its own location, so every entry point runs from a
  fresh clone with no `PYTHONPATH` setting and no installation step.

## Validation performed for this release

Checks run on the released repository: byte-compilation of every Python file,
an AST scan resolving every internal import, YAML parsing of every
configuration, a check that every README command names an existing script and
flag, a test that `scripts/make_splits.py` reproduces the archived split from
the archived slice records, and an AST-level comparison of every model, loss,
metric, dataset, trainer and augmentation function against the archived
experiment code (differences were only renames, refactors and the items
above). The sandbox had no PyTorch, timm, Flower or SimpleITK and no FedBCa
data, so no model was instantiated and no training, evaluation or federated
run was executed.
