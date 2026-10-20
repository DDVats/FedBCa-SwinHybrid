# Reproducibility

All values below are taken from the final paper. Read this together with
`docs/PAPER_CODE_AUDIT.md`, which lists the points where the released code and
the published description differ, and `docs/DATASET.md`, which explains how to
obtain the data.

## Hardware reported in the paper

* NVIDIA GeForce RTX 4070, 12 GB VRAM

The 12 GB budget is what sets the batch configuration: batch size 6 with four
gradient accumulation steps, giving an effective batch of 24. On a smaller
card, lower `batch_size` and raise `accum_steps` by the same factor to keep the
effective batch at 24. Federated training runs four clients; on one machine
they share the GPU, which is why `num_workers` defaults to 0 in
`configs/federated.yaml`.

## Software

* PyTorch >= 2.1
* timm 0.9.16 (ImageNet-pretrained ResNet-34 and Swin-Tiny weights)
* Flower (federated learning)
* SimpleITK (NIfTI reading, resampling, optional N4 correction)

See `requirements.txt` for the full list. CUDA versions are deliberately not
pinned: install the PyTorch build that matches your driver.

## Preprocessing

| Step | Setting |
|---|---|
| N4 bias-field correction | disabled by default, optional (audit item 1) |
| In-plane resampling | 0.8 mm x 0.8 mm, B-spline |
| Intensity clipping | foreground (5th-percentile threshold), 0.5th-99.5th percentile |
| Normalisation | Z-score, clipped to [-5, 5] |
| 2.5D stack | target slice +/- 2 slices, 5 channels |
| Crop | tumour-centred ROI, 40 px padding |
| Model input | 224 x 224 (audit item 5) |

## Data partitioning

Patient-level stratified split performed independently within each centre,
preserving the MIBC/NMIBC balance: approximately 71% training, 10%
validation, 19% test. The same partitions are used for the centralised and
the federated experiments, and no patient contributes to more than one split.

## Centralised training

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Schedule | warmup-cosine |
| Gradient clipping | max norm 1.0 |
| Base learning rate | 2 x 10^-5 |
| ResNet-34 LR multiplier | 1.0 in the recorded runs (paper text: 0.15; audit item 6) |
| Swin-Tiny LR multiplier | 0.30 |
| Weight decay | 1 x 10^-4 |
| Effective batch size | 24 (batch 6 x 4 accumulation steps) |
| Precision | FP32 |
| Loss | `L = 0.80 L_seg + 0.20 L_cls` |
| Classification ramp | weight reaches its full value at cumulative epoch 20 (centralised only; the federated client uses a fixed 0.20, audit item 8) |

```bash
python scripts/train.py --config configs/swin_hybrid.yaml \
    --augmentation geo_elastic
```

## Federated training

| Setting | Value |
|---|---|
| Clients | 4, one per clinical centre |
| Participation | all four clients in every round |
| Local epochs per round | 3 |
| Communication rounds | 50 |
| Aggregation | sample-size-weighted FedAvg |
| Augmentation | Geo+Elastic |

```bash
python scripts/federated_server.py --config configs/federated.yaml
# then, one terminal per centre
python scripts/federated_client.py --center 1
python scripts/federated_client.py --center 2
python scripts/federated_client.py --center 3
python scripts/federated_client.py --center 4
```

## Augmentation (Geo+Elastic)

* random horizontal flip, p = 0.5
* rotation within +/- 15 degrees, p = 0.70
* elastic deformation, sigma = 5.0, alpha = 18.0, p = 0.20

## Model selection

The retained model is the one with the highest validation combined score

```
S_comb = 0.55 * DSC + 0.45 * AUC
```

where DSC is computed on slices with a non-empty ground-truth mask and AUC is
computed at the patient level after averaging the slice-level MIBC
probabilities. The combined score is used for selection only; DSC and AUC are
reported separately. Centrally, selection switches from training loss to the
combined score once the classification loss has finished ramping in, so that
early epochs with an uninformative AUC cannot win selection. In the federated
setting the server retains the global model with the best sample-weighted
combined score across the four centres.

## Seeds and determinism

Scripts seed Python, NumPy and PyTorch from the `seed` key (default 42); each
federated client offsets that seed by its centre index. This does not make runs
bit-identical. Dataloader workers, cuDNN kernel selection and GPU atomics all
introduce non-determinism, and no deterministic-algorithm flag is forced,
because doing so would change the training speed and the kernels relative to
the reported runs.

## What can and cannot be reproduced here

The repository is code-ready for GitHub publication: the method, the training
protocol and the evaluation are fully specified and runnable.

It is **not** self-contained for reproduction. FedBCa is not distributed here,
so none of the published numbers can be regenerated from this repository
alone. With the dataset in place, expect results near the published ones
rather than identical to them, because the patient-level split is redrawn, GPU
training is non-deterministic, and the audit items (notably the ResNet-34 learning-rate multiplier, item 6)
mean the released defaults do not exactly match the configuration that
produced the reported runs. No trained checkpoints are distributed.
