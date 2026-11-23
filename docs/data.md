# Data provenance and setup

## What was recoverable

The repository history contains three commits: `b5e39e6`, `01a552d`, and `54645cf`. The first commit already refers to `./data_seg_public.zip`, with commented download code that uses an undefined `URL`. Neither the archive, a real download URL, trained weights, nor numerical experiment results appears in those commits. A check of unreachable Git objects did not reveal additional artifacts. The old README identifies Cityscapes and links two qualitative images; those images alone cannot establish a reproducible benchmark.

The eight RGB colors in the original loader indicate a category-level derivative, rather than the full original Cityscapes color taxonomy. The exact archive's contents and preparation process cannot be verified from the repository. Exact-name web searches did not locate a verified download for `data_seg_public.zip`.

## Official Cityscapes route

The [official download page](https://www.cityscapes-dataset.com/downloads/) requires a Cityscapes account. Obtain `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` through that account and extract both under `data/cityscapes/`. The archive names and layout are also documented by [NVIDIA's dataset preparation guide](https://github.com/NVIDIA/semantic-segmentation/blob/main/PREPARE_DATASETS.md).

The resulting top-level directories must be `leftImg8bit` and `gtFine`, not separate archive-name wrappers. Training uses `train` and `val` only. Official test annotations do not provide the public semantic ground truth needed for this local evaluation workflow; use a separately labeled holdout if you need an independent final score.

Use `configs/cityscapes.json`. It reads `*_gtFine_labelIds.png`, keeps the official evaluable labels, and maps them into this project's categories. It does **not** accept `labelTrainIds` as original IDs: those numbers have different meanings. Preserve the original filenames to help prevent this mistake.

No download credentials or dataset files should be committed. Dataset access and redistribution follow the provider's terms. The project does not attempt to reconstruct a private download URL.

## Simplified dataset route

If the original archive is recovered, extract it manually. Both layouts work:

```text
DATA/data_seg_public/img/<city>/*_leftImg8bit.png
DATA/data_seg_public/mask/<city>/*_gtFine_color.png
```

```text
DATA/img/train/<optional-group>/sample.png
DATA/mask/train/<optional-group>/sample.png
DATA/img/val/<optional-group>/sample.png
DATA/mask/val/<optional-group>/sample.png
```

Generic images and masks have matching relative PNG paths. Cityscapes-style image names are paired with the corresponding `gtFine` name. For `category_ids`, the corresponding Cityscapes-style suffix is `_gtFine_categoryIds.png`.

| Encoding | Expected mask | Ignored pixels |
| --- | --- | --- |
| `category_rgb` | Exactly the palette below | Void by default; unknown colors raise an error |
| `category_ids` | Single-channel IDs 0–7 | 255, plus void by default |
| `cityscapes_ids` | Original Cityscapes label IDs 0–33 / 255 | Official `ignoreInEval` labels, including void |

| Category ID | Name | RGB |
| ---: | --- | --- |
| 0 | void | 0, 0, 0 |
| 1 | flat | 128, 64, 128 |
| 2 | construction | 70, 70, 70 |
| 3 | object | 153, 153, 153 |
| 4 | nature | 107, 142, 35 |
| 5 | sky | 70, 130, 180 |
| 6 | human | 220, 20, 60 |
| 7 | vehicle | 0, 0, 142 |

Do not pass original Cityscapes RGB masks to `category_rgb`: original colors include distinctions such as sidewalk, rider, and truck that are absent from this reduced palette. The strict loader makes this mismatch visible instead of silently treating pixels as void.

Images are converted to RGB, resized over the full frame, and normalized using the constants in `dataset.py`. Masks are decoded before nearest-neighbor resizing. Training can apply a paired horizontal flip and image-only brightness/contrast changes. Validation never applies random augmentation. Resizing an image with a different aspect ratio to the configured dimensions distorts its geometry; choose a suitable resolution for a different dataset.

## Using a remote GPU

[Google Colab](https://research.google.com/colaboratory/faq.html) provides GPU runtimes subject to availability and changing usage limits. A free GPU is not guaranteed. Once these changes are pushed to your repository, clone that revision in a notebook, select a GPU runtime, and install the project:

```python
!git clone https://github.com/pailozka/Image-Segmentation.git
%cd Image-Segmentation
%pip install -e '.[plots]'

import torch
print(torch.__version__, torch.cuda.is_available())
```

Place the legitimately downloaded Cityscapes files under the documented layout on the runtime disk. Mount persistent storage for run artifacts; for example, in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!python training.py /content/data/cityscapes --config configs/cityscapes.json --output /content/drive/MyDrive/segmentation-runs/attention-seed42 --device cuda --workers 2
```

If the runtime disconnects, restore the same project revision, libraries, data path, configuration, and worker count, then pass `--resume /content/drive/MyDrive/segmentation-runs/attention-seed42/last.pt` with the same output directory. The checkpoint resumes from the next completed-epoch boundary. A different GPU/library combination may change results even when seeds match.

The included synthetic data generator permits local CPU verification while real data or GPU access is unavailable. It is intentionally labeled as fixture data.
