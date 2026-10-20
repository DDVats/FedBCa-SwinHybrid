# Dataset

> This repository contains research code only. It does not distribute medical
> data and does not provide clinical diagnosis. The FedBCa dataset must be
> obtained separately and used according to its applicable terms.

## What FedBCa is

FedBCa is a multicentre bladder cancer MRI dataset of **275 T2-weighted MRI
scans** collected at **four medical centres**. Each scan carries a pixel-level
bladder tumour segmentation mask and a pathological **MIBC / NMIBC** label
(muscle-invasive versus non-muscle-invasive bladder cancer). Because the four
centres differ in scanner hardware, field strength and acquisition protocol,
the dataset is naturally non-IID, which is what makes it suitable for
evaluating federated learning.

## Centre distribution

As reported in Table 1 of the paper:

| Centre | Scans | NMIBC | MIBC |
|---|---|---|---|
| Center 1 | 160 | 130 | 30 |
| Center 2 | 48 | 23 | 25 |
| Center 3 | 32 | 19 | 13 |
| Center 4 | 35 | 18 | 17 |
| **Total** | **275** | **190** | **85** |

The MIBC share varies from about 19% at Centre 1 to about 52% at Centre 2.
This label skew across clients is part of the problem setting, not an artefact
of the split.

## Obtaining the data

The dataset accompanies:

> Cao, K., Zou, Y., Zhang, C., Zhang, W., Zhang, J., Wang, G., Zhang, C.,
> Lyu, J., Sun, Y., Zhang, H., et al. *A multicenter bladder cancer MRI
> dataset and baseline evaluation of federated learning in clinical
> application.* Scientific Data **11**(1), 1147 (2024).

Please obtain the FedBCa dataset from its official publication/source and
follow its access and licensing requirements. No download link is reproduced
here, and this repository does not mirror or redistribute any part of the
data.

## Expected directory layout

`scripts/preprocess.py` expects one directory per centre below the dataset
root:

```
$FEDBCA_ROOT/
├── Center1/
│   ├── Image/                 # T2-weighted volumes (NIfTI)
│   ├── Annotation/            # tumour segmentation masks (NIfTI)
│   └── Center_1_label.xlsx    # per-scan MIBC / NMIBC labels
├── Center2/
│   ├── Image/
│   ├── Annotation/
│   └── Center_2_label.xlsx
├── Center3/
│   └── ...
└── Center4/
    └── ...
```

The reader is tolerant of the naming variants seen in distributions of this
dataset. Image directories may be named `Image`, `T2WI`, `image` or `images`;
mask directories `Annotation`, `annotation`, `Mask` or `mask`; label files
`Center_<i>_label.xlsx`, `center<i>_label.xlsx`, `label_<i>.xlsx` or
`label.xlsx`. Within the spreadsheet, the image column may be `image`,
`image_name`, `filename` or `file`; the mask column `mask_new`, `annotation`,
`mask`, `mask_name` or `seg`; the label column `MIBC`, `mibc`, `label` or
`class`. If your copy uses different names, extend `COLUMN_CANDIDATES` in
`scripts/preprocess.py` rather than renaming the data.

## Pointing the code at your copy

Nothing is hard-coded. Set the environment variables, or pass the equivalent
command-line flags:

```bash
export FEDBCA_ROOT=/path/to/FedBCa              # your local copy
export FEDBCA_PREPROCESSED=/path/to/preprocessed # written by preprocess.py
export FEDBCA_SPLITS=/path/to/splits             # written by make_splits.py
export FEDBCA_OUTPUT=/path/to/outputs            # checkpoints and logs
```

## What preprocessing produces

```bash
python scripts/preprocess.py --config configs/preprocessing.yaml
python scripts/make_splits.py \
    --slice-index $FEDBCA_PREPROCESSED/slice_index.csv \
    --output-dir $FEDBCA_SPLITS
```

Per retained slice, two files are written:

* `<patient_id>_s<NNN>_img.npy` - the 2.5D stack, `(5, 224, 224)` float32;
* `<patient_id>_s<NNN>_mask.npy` - the tumour mask, `(224, 224)` uint8.

Alongside them, `slice_index.csv` lists every slice with its patient id,
centre, MIBC/NMIBC label, tumour flag and largest-tumour-area flag.
`make_splits.py` turns that index into `train.csv`, `val.csv` and `test.csv`,
split at the patient level, stratified by class and performed independently
within each centre, so that no patient appears in more than one split.

Keep these outputs outside the repository. The `.gitignore` already excludes
`data/`, `outputs/` and `*.npy`, but the surest safeguard is to point
`FEDBCA_PREPROCESSED` somewhere outside the working tree.

## Privacy

Patient data, derived slice tensors, predicted masks, split files containing
patient identifiers and trained checkpoints are all out of scope for this
repository and none are included in it. If you fork this work, check that no
such file has been committed before pushing.
