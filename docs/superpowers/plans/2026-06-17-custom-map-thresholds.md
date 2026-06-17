# Custom mAP Thresholds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add evaluate output for mAP50, mAP80, mAP85, mAP90, and mAP95 without reporting mAP50-95 in the custom metrics file.

**Architecture:** Keep Ultralytics evaluation unchanged, then post-process the returned metrics object. A small `eval_metrics` module will extract AP values from `result.box.all_ap`, combine them with precision/recall when available, and write `custom_metrics.csv` under the evaluation run directory.

**Tech Stack:** Python standard library, `unittest`, existing YOLO11-OBB scripts.

---

### Task 1: Metric Extraction

**Files:**
- Create: `tests/test_eval_metrics.py`
- Create: `yolo11_obb/eval_metrics.py`

- [ ] Write a failing unittest using a fake Ultralytics result object with `box.all_ap`, `box.p`, `box.r`, and class names.
- [ ] Implement extraction of IoU threshold indices: 0.50 -> 0, 0.80 -> 6, 0.85 -> 7, 0.90 -> 8, 0.95 -> 9.
- [ ] Produce an `all` row as the mean over classes and per-class rows using class names.
- [ ] Write `custom_metrics.csv` with columns `class,precision,recall,mAP50,mAP80,mAP85,mAP90,mAP95`.

### Task 2: Evaluate Script Integration

**Files:**
- Modify: `scripts/evaluate_yolo11_obb.py`
- Modify: `README.md`

- [ ] Add a call to write the custom metrics CSV after `evaluate(options)` returns.
- [ ] Print the custom metrics path.
- [ ] Document the new evaluate output and keep training/evaluate commands unchanged.

### Task 3: Verification

**Files:**
- Test: `tests/test_eval_metrics.py`

- [ ] Run `python3 -m unittest tests/test_eval_metrics.py`.
- [ ] Run `python3 -m unittest discover -s tests`.
- [ ] Run `python3 scripts/evaluate_yolo11_obb.py --help` to verify CLI still loads without Ultralytics.
