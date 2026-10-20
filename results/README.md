# Results

Every number in this directory is transcribed from the final paper
(Tables 2 and 3). Nothing here was regenerated or recomputed, and no result
file from the development workspace was carried over, because those files
disagree with each other and in places with the paper; see item 9 of
`docs/PAPER_CODE_AUDIT.md`.

## `architecture_comparison.csv` (Table 2)

Centralised comparison of the five candidate architectures, on the same
patient-level partitions and metrics.

| Model | DSC | AUC | Combined |
|---|---|---|---|
| ResNet50 U-Net (single-task) | **0.8181** | 0.8304 | 0.8236 |
| EfficientNet-B0 U-Net (single-task) | 0.7669 | 0.7589 | 0.7633 |
| ResNet50+CBAM (multi-task) | 0.7579 | 0.7232 | 0.7423 |
| Multi-task EfficientNet-B0 | 0.7653 | 0.7679 | 0.7664 |
| **Swin Hybrid (proposed)** | 0.8088 | **0.9286** | **0.8627** |

The Swin Hybrid has the best AUC and combined score and was selected for the
augmentation study. The single-task ResNet50 U-Net reaches a higher DSC, but it
does not produce an MIBC prediction from a shared representation.

## `augmentation_comparison.csv` (Table 3)

The seven augmentation strategies, each evaluated with the Swin Hybrid model
under centralised and under federated training.

| Augmentation | Cen. DSC | Cen. AUC | Cen. Comb. | Fed. DSC | Fed. AUC | Fed. Comb. |
|---|---|---|---|---|---|---|
| None | 0.8088 | **0.9286** | 0.8627 | 0.7908 | 0.7716 | 0.7822 |
| GIN Only | 0.8161 | 0.8750 | 0.8426 | 0.7790 | 0.7270 | 0.7560 |
| Augmentation Stack | **0.8487** | 0.8661 | 0.8565 | **0.8220** | 0.8230 | 0.8225 |
| **Geo+Elastic** | 0.8272 | 0.9196 | **0.8688** | 0.8100 | **0.8931** | **0.8474** |
| Geo+Bias | 0.8391 | 0.8661 | 0.8513 | 0.8094 | 0.8414 | 0.8238 |
| Geo+Gamma | 0.8417 | 0.8750 | 0.8567 | 0.7987 | 0.8575 | 0.8252 |
| Geo+Noise | 0.8398 | 0.8750 | 0.8556 | 0.8108 | 0.8240 | 0.8167 |

Geo+Elastic has the highest combined score in both settings and is the final
strategy of the paper. The Augmentation Stack reaches the highest DSC in both
settings, but its lower AUC leaves it behind on the combined score.

## Reproducing these numbers

They cannot be reproduced from this repository alone: FedBCa is not
distributed here and must be obtained from its own source (see
`docs/DATASET.md`). With the dataset in place, `docs/REPRODUCIBILITY.md`
gives the full training protocol. Differences in the shuffled patient-level
split, in GPU non-determinism and in library versions mean that a re-run will
land near these values rather than on them.

Known differences between the published protocol and the released code are
listed in `docs/PAPER_CODE_AUDIT.md`.

## Score definitions

* **DSC** - Dice Similarity Coefficient, computed only on slices with a
  non-empty ground-truth tumour mask.
* **AUC** - area under the ROC curve at the patient level, after averaging
  the slice-level MIBC probabilities of each patient.
* **Combined** - `0.55 * DSC + 0.45 * AUC`, used for model selection.
