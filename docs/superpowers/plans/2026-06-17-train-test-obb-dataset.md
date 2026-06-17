# Train Test OBB Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a label1-label6 YOLO-OBB dataset from the converted AnyLabeling OBB folder with train/test split only.

**Architecture:** Add a focused dataset splitter module that reads a flat OBB image/txt directory, groups sibling crop images by parent sample, filters class IDs to 0-5, and writes `images/train`, `labels/train`, `images/test`, `labels/test`, plus `data.yaml` and split reports. Update config and training wrappers so train/test-only datasets validate locally and training can disable epoch validation with `--no-val`.

**Tech Stack:** Python standard library, `unittest`, existing `yolo11_obb` helpers and script conventions.

---

### Task 1: Dataset Splitter

**Files:**
- Create: `yolo11_obb/dataset_splitter.py`
- Create: `scripts/create_train_test_dataset.py`
- Test: `tests/test_dataset_splitter.py`

- [ ] Write failing tests for grouped train/test split, class filtering, and `data.yaml` with no dedicated val directory.
- [ ] Implement group-key extraction by dropping the final crop suffix such as `-1` from `CropImage_..._NG-1`.
- [ ] Implement deterministic group shuffle with seed 42 and train ratio 0.8.
- [ ] Copy images and filtered labels into `images/{train,test}` and `labels/{train,test}`.
- [ ] Write `split_manifest.csv`, `split_report.txt`, and `data.yaml`.
- [ ] Run `python3 -m unittest tests/test_dataset_splitter.py`.

### Task 2: Train/Test Config Support

**Files:**
- Modify: `yolo11_obb/config.py`
- Modify: `scripts/train_yolo11_obb.py`
- Test: `tests/test_yolo11_obb_config.py`

- [ ] Write failing tests proving `data.yaml` can omit `val`, layout validation accepts only declared splits, and train kwargs can include `val=False`.
- [ ] Update `load_dataset_config`, `validate_dataset_layout`, and `format_layout_report` to use declared split keys.
- [ ] Add `validate_during_training` to `TrainOptions` and map it to Ultralytics `val`.
- [ ] Add `--no-val` to the train script.
- [ ] Run relevant tests and full suite.

### Task 3: Generate Dataset

**Files:**
- Output: `datasets/154843_obb_converted_label1_6_train_test`

- [ ] Run `python3 scripts/create_train_test_dataset.py`.
- [ ] Verify 429 images are split approximately 80/20 by grouped parent sample, only classes 0-5 remain, every label has 9 fields, and all coordinates are in `[0,1]`.
