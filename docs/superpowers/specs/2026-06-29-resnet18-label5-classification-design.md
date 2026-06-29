# ResNet18 Label5 Classification Design

## Goal

Build the first classification training path after YOLO11-OBB detection by training a ResNet18 classifier on cropped `label5` regions. The first version uses manually annotated AnyLabeling OBB boxes for crop generation, and uses `outputs/label1_6_description.xlsx` as the classification label source.

## Current Data Findings

The filtered Excel workbook covers samples after timestamp `20260121210219803` and excludes trailing image index `1`.

Per-label class distributions from `outputs/label1_6_description.xlsx`:

| label | target field | distribution | suitability |
| --- | --- | --- | --- |
| `label1` | `tag1` | `OK=251`, `NG=0` | not usable for OK/NG training yet |
| `label1` | `tag2` | `B=127`, `G=43`, `R=40`, `W=37`, empty `4` | usable later as color classification, moderately imbalanced |
| `label2` | `tag1` | `OK=251`, `NG=0` | not usable for binary training |
| `label3` | `tag1` | `OK=86`, `NG=165` | usable, but less balanced than label5 |
| `label4` | `tag1` | `OK=250`, `OTHER=1` | not useful for first classifier |
| `label5` | `tag1` | `OK=98`, `NG=153` | best first binary target |
| `label6` | `tag1` | `OK=244`, `NG=3`, empty `4` | too imbalanced |

The first classifier should therefore target `label5` `OK/NG`, because it has the most balanced useful binary distribution in the current labeled data.

## Label1 Multi-Task Decision

`label1` should eventually use a multi-task classifier:

```text
ResNet18 backbone
  status_head: OK / NG
  color_head: B / G / R / W
```

Training should mask `color_head` loss for samples without a color label or for future samples where color is not applicable. However, the current filtered data has no `label1 NG` samples, so training a real `label1` OK/NG head now would produce a degenerate model that always predicts `OK`. For the first phase, `label1` is out of scope except as a future design constraint.

## First-Phase Scope

The first implementation should create a full but narrow classification path for `label5`:

1. Read `outputs/label1_6_description.xlsx`.
2. Convert the `label5` sheet into a CSV manifest with `image_name`, `class_name`, and split metadata.
3. Read the matching AnyLabeling JSON from `已打标的数据202604/user1_2026-03-16_154843_anylabeling`.
4. Locate the `label5` polygon in each JSON.
5. Crop the `label5` OBB region from the original image using the annotated polygon.
6. Split by parent image group at `8:2`, reusing the repo convention that stems ending in `-N` share a parent group.
7. Train a ResNet18 binary classifier on the generated crops.
8. Evaluate on the held-out test split with accuracy, per-class precision, per-class recall, F1, confusion matrix counts, and a CSV of per-image predictions.

This phase does not use YOLO-predicted boxes for training crops. YOLO-predicted boxes are a later robustness phase after the clean supervised classification path works.

## Data Flow

```text
outputs/label1_6_description.xlsx
  -> label5 manifest rows
  -> group-aware 8:2 train/test split
  -> CSV manifest

AnyLabeling JSON + source BMP image
  -> label5 polygon
  -> perspective crop / rectified crop
  -> datasets/classification/label5_ok_ng/images/{train,test}/{OK,NG}/...

ResNet18 training script
  -> runs/classification/label5_resnet18_*/weights/best.pt
  -> metrics.csv
  -> predictions.csv
  -> confusion_matrix.csv
```

## Crop Strategy

Use the annotated four-point polygon for `label5`, not the axis-aligned bounding rectangle. The cropper should rectify the OBB polygon with a perspective transform so the classifier sees a consistent upright rectangular region. The output crop should be saved as a standard image file such as `.png`.

If the polygon is invalid, the source image is missing, or the Excel row has no matching JSON, the dataset builder should fail with a clear file-specific error instead of silently skipping the sample.

## Split Strategy

Use an 8:2 split by parent group, not row-level random splitting. For a stem like:

```text
CropImage_20260126092714769_F3-I0_OK-5
```

the parent group is:

```text
CropImage_20260126092714769_F3-I0_OK
```

All child indices from the same parent group must go into the same split. This avoids leaking near-duplicate related samples across train and test.

## Model Design

The first training script should use `torchvision.models.resnet18`.

Recommended defaults:

| setting | value |
| --- | --- |
| input size | `224x224` |
| pretrained weights | ImageNet pretrained when available |
| classifier head | 2 output classes: `OK`, `NG` |
| optimizer | AdamW |
| loss | cross entropy, with optional class weighting |
| epochs | configurable, default `30` |
| batch size | configurable, default `32` |
| split | prebuilt train/test folders from the dataset builder |

The code should not assume CUDA exists. It should choose CUDA when available, otherwise CPU, unless the user passes an explicit device. ImageNet pretrained weights should be configurable because a server without network or cached weights must still be able to run with random initialization.

Because the requested split is only `8:2` train/test, the first version may select `best.pt` by test macro F1 or test accuracy. The training output must make this explicit in `args.yaml` and logs so the reported test metric is understood as optimistic model-selection feedback, not a final untouched benchmark.

## Files To Add

The implementation plan should create focused modules rather than one large script:

| file | responsibility |
| --- | --- |
| `yolo11_obb/classification_labels.py` | parse Excel sheets, normalize `OK/NG`, compute parent groups |
| `yolo11_obb/obb_crop.py` | rectify and save OBB crops from four-point polygons |
| `yolo11_obb/classification_dataset.py` | build crop folders and CSV manifests for one label |
| `scripts/create_label_classification_dataset.py` | CLI for creating a classification dataset |
| `scripts/train_resnet18_classifier.py` | CLI for ResNet18 training/evaluation |
| `tests/test_classification_labels.py` | unit tests for Excel parsing and split grouping |
| `tests/test_obb_crop.py` | unit tests for crop geometry |
| `tests/test_classification_dataset.py` | unit tests for manifest generation and dataset layout |

## Outputs

Dataset builder output should be under:

```text
datasets/classification/label5_ok_ng/
```

Training output should be under:

```text
runs/classification/label5_resnet18_<run_name>/
```

The dataset directory should include:

| file or directory | purpose |
| --- | --- |
| `images/train/OK/` | train crops for OK |
| `images/train/NG/` | train crops for NG |
| `images/test/OK/` | test crops for OK |
| `images/test/NG/` | test crops for NG |
| `manifest.csv` | all samples with source paths, labels, split, and crop path |
| `split_report.txt` | counts by split and class |

The training directory should include:

| file | purpose |
| --- | --- |
| `weights/best.pt` | best checkpoint by test accuracy or F1 |
| `weights/last.pt` | last checkpoint |
| `metrics.csv` | epoch-level train/test metrics |
| `predictions.csv` | per-sample predictions on test split |
| `confusion_matrix.csv` | counts for `OK/NG` predictions |
| `args.yaml` | training configuration for reproducibility |

## Verification

The implementation should be verified at three levels:

1. Unit tests for Excel parsing, group split, and OBB crop geometry.
2. A dry-run dataset build on small fixture data.
3. A real dataset build from `outputs/label1_6_description.xlsx`, followed by a short smoke training run with very few epochs to confirm that the training script, dataloaders, metrics, and checkpoint writing work.

Full training on the server can use the same scripts with the server-side YOLO weights later when predicted-box classification is introduced.

## Out Of Scope For First Phase

- Training `label1` multi-task classification.
- Using YOLO-predicted boxes as crop sources.
- Training one combined classifier for all label regions.
- Updating the detection model.
- Running a full server training job from the local workspace.
