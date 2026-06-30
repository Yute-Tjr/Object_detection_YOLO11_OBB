# YOLO ResNet Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prediction pipeline that runs YOLO11-OBB detection, classifies detected `label3` and `label5` crops with ResNet18, and writes CSV plus visualization outputs.

**Architecture:** Add a focused `yolo11_obb/pipeline_predict.py` module for data classes, result parsing, crop classification, summary rules, CSV writing, and visualization. Add a thin CLI at `scripts/predict_yolo11_resnet_pipeline.py`. Reuse existing OBB crop and ResNet18 inference helpers.

**Tech Stack:** Python, Ultralytics YOLO, OpenCV, PyTorch, torchvision, CSV.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `yolo11_obb/pipeline_predict.py` | End-to-end pipeline helpers and orchestration. |
| `scripts/predict_yolo11_resnet_pipeline.py` | CLI argument parsing and path resolution. |
| `tests/test_pipeline_predict.py` | Unit tests for pure pipeline behavior and visualization output. |
| `README.md` | Optional command documentation after implementation. |

## Task 1: Pure Pipeline Data Rules

**Files:**
- Create: `tests/test_pipeline_predict.py`
- Create: `yolo11_obb/pipeline_predict.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
from pathlib import Path

from yolo11_obb.pipeline_predict import (
    PipelineDetection,
    final_result_from_selected,
    format_detection_row,
    selected_detection_by_label,
    summary_row_for_image,
)


def test_selected_detection_by_label_chooses_highest_confidence():
    detections = [
        PipelineDetection(Path("a.png"), "a.png", 0, "label3", 0.30, ((0, 0), (1, 0), (1, 1), (0, 1))),
        PipelineDetection(Path("a.png"), "a.png", 1, "label3", 0.90, ((2, 2), (3, 2), (3, 3), (2, 3))),
    ]
    selected, warnings = selected_detection_by_label(detections, ["label3", "label5"])
    assert selected["label3"].det_index == 1
    assert "duplicate label3 count=2" in warnings
    assert "missing label5" in warnings


def test_final_result_from_selected_marks_ng_ok_and_unknown():
    assert final_result_from_selected({"label3": "OK", "label5": "OK"}) == "OK"
    assert final_result_from_selected({"label3": "NG", "label5": "OK"}) == "NG"
    assert final_result_from_selected({"label3": "OK"}) == "UNKNOWN"


def test_format_detection_row_includes_empty_classification_fields():
    detection = PipelineDetection(Path("a.png"), "a.png", 0, "label2", 0.75, ((0, 0), (1, 0), (1, 1), (0, 1)))
    row = format_detection_row(detection)
    assert row["det_label"] == "label2"
    assert row["classifier"] == ""
    assert row["prob_NG"] == ""


def test_summary_row_for_image_uses_selected_predictions():
    label3 = PipelineDetection(Path("a.png"), "a.png", 0, "label3", 0.80, ((0, 0), (1, 0), (1, 1), (0, 1)))
    label3.predicted_label = "OK"
    label3.cls_confidence = 0.90
    label5 = PipelineDetection(Path("a.png"), "a.png", 1, "label5", 0.70, ((0, 0), (1, 0), (1, 1), (0, 1)))
    label5.predicted_label = "NG"
    label5.cls_confidence = 0.95
    row = summary_row_for_image(Path("a.png"), [label3, label5], Path("vis/a.png"))
    assert row["final_result"] == "NG"
    assert row["label3_pred"] == "OK"
    assert row["label5_pred"] == "NG"
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv/bin/python tests/test_pipeline_predict.py
```

Expected: fail with `ModuleNotFoundError` or missing function imports.

- [ ] **Step 3: Implement data classes and pure helpers**

Implement:

```python
@dataclass
class PipelineDetection:
    image_path: Path
    image_name: str
    det_index: int
    det_label: str
    det_conf: float
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    crop_path: str = ""
    classifier: str = ""
    predicted_label: str = ""
    cls_confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
```

Also implement:

- `selected_detection_by_label`
- `final_result_from_selected`
- `format_detection_row`
- `summary_row_for_image`

- [ ] **Step 4: Verify green**

Run:

```bash
.venv/bin/python tests/test_pipeline_predict.py
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add yolo11_obb/pipeline_predict.py tests/test_pipeline_predict.py
git commit -m "添加检测分类流程基础规则"
```

## Task 2: Visualization

**Files:**
- Modify: `tests/test_pipeline_predict.py`
- Modify: `yolo11_obb/pipeline_predict.py`

- [ ] **Step 1: Write failing visualization test**

Add a test that creates a small black image, draws one `label3` `NG` detection, and asserts the output image exists and has non-zero pixels:

```python
def test_draw_visualization_writes_annotated_image(tmp_path):
    image_path = tmp_path / "a.png"
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    detection = PipelineDetection(image_path, image_path.name, 0, "label3", 0.9, ((10, 10), (60, 10), (60, 40), (10, 40)))
    detection.predicted_label = "NG"
    detection.cls_confidence = 0.95
    output = tmp_path / "vis.png"
    draw_visualization(image_path, [detection], output)
    saved = cv2.imread(str(output))
    assert saved is not None
    assert int(saved.sum()) > 0
```

- [ ] **Step 2: Verify red**

Run `.venv/bin/python tests/test_pipeline_predict.py`; expected missing `draw_visualization`.

- [ ] **Step 3: Implement visualization**

Use OpenCV `polylines` and `putText`; colors:

- NG: red `(0, 0, 255)`
- OK: green `(0, 180, 0)`
- unknown/unclassified: yellow `(0, 200, 255)`

- [ ] **Step 4: Verify green**

Run `.venv/bin/python tests/test_pipeline_predict.py`; expected `OK`.

- [ ] **Step 5: Commit**

```bash
git add yolo11_obb/pipeline_predict.py tests/test_pipeline_predict.py
git commit -m "添加流程可视化绘制"
```

## Task 3: Orchestration And CLI

**Files:**
- Modify: `yolo11_obb/pipeline_predict.py`
- Create: `scripts/predict_yolo11_resnet_pipeline.py`

- [ ] **Step 1: Implement YOLO result parsing**

Add `detections_from_yolo_result(result)`. It reads:

- `result.path`
- `result.names`
- `result.obb.cls`
- `result.obb.conf`
- `result.obb.xyxyxyxy`

Skip empty OBB results.

- [ ] **Step 2: Implement classifier loading and crop classification**

Add:

- `PipelineClassifier`
- `load_pipeline_classifiers`
- `classify_detection_crop`

Use `load_resnet18_checkpoint`, `classification_transform`, and `rectify_obb_crop`.

- [ ] **Step 3: Implement `run_pipeline`**

Run YOLO with:

```python
model.predict(source=str(source), imgsz=imgsz, conf=det_conf, device=det_device, save=False, verbose=True)
```

Write:

- `detections.csv`
- `summary.csv`
- `pipeline_report.txt`
- crops
- visualizations

- [ ] **Step 4: Add CLI**

Create `scripts/predict_yolo11_resnet_pipeline.py` with arguments:

- `--det-weights`
- `--source`
- `--label3-weights`
- `--label5-weights`
- `--imgsz`
- `--det-conf`
- `--det-device`
- `--cls-device`
- `--project`
- `--name`
- `--exist-ok`

- [ ] **Step 5: Run import/compile checks**

```bash
.venv/bin/python -m py_compile yolo11_obb/pipeline_predict.py scripts/predict_yolo11_resnet_pipeline.py
```

- [ ] **Step 6: Commit**

```bash
git add yolo11_obb/pipeline_predict.py scripts/predict_yolo11_resnet_pipeline.py
git commit -m "接入目标检测和分类预测流程"
```

## Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README command**

Document:

```bash
python3 scripts/predict_yolo11_resnet_pipeline.py \
  --det-weights runs/obb/your_detector/weights/best.pt \
  --source path/to/images \
  --label3-weights runs/classification/label3_resnet18_cpu_e30/weights/best.pt \
  --label5-weights runs/classification/label5_resnet18_mps_e30/weights/best.pt \
  --imgsz 1280 \
  --det-conf 0.25 \
  --det-device 0 \
  --cls-device 0 \
  --name yolo11_resnet_pipeline
```

- [ ] **Step 2: Run targeted tests**

```bash
.venv/bin/python tests/test_pipeline_predict.py
.venv/bin/python tests/test_obb_crop.py
.venv/bin/python tests/test_classification_inference.py
```

- [ ] **Step 3: Verify git status**

Only source/doc/test files should be tracked. Generated `runs/pipeline/` outputs stay ignored by `/runs/`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "补充检测分类联动预测说明"
```
