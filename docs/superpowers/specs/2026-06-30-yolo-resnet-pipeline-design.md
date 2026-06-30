# YOLO11-OBB + ResNet18 Pipeline Design

## Goal

Build an end-to-end prediction pipeline that connects the trained YOLO11-OBB detector with the trained ResNet18 classifiers for `label3` and `label5`, then writes CSV outputs, OBB crops, and annotated visualization images.

## Inputs

- Detection weights: YOLO11-OBB `best.pt`.
- Source: one image or a directory of images.
- Classification weights:
  - `label3`: ResNet18 OK/NG classifier checkpoint.
  - `label5`: ResNet18 OK/NG classifier checkpoint.
- Runtime options: detection image size, detection confidence, detection device, classification device, output project/name.

## Data Flow

1. Collect input images from the source path.
2. Run YOLO11-OBB prediction with `save=False`; parse each result's OBB polygons, class ids, confidences, and class names.
3. Keep all detected labels in `detections.csv`.
4. For detections whose label is registered to a classifier (`label3`, `label5`):
   - Read the original image with OpenCV.
   - Rectify the OBB polygon using `yolo11_obb.obb_crop.rectify_obb_crop`.
   - Save the crop under `crops/<image_stem>/<label>_<index>.png`.
   - Run the matching ResNet18 classifier on the crop.
   - Store predicted label, confidence, and per-class probabilities.
5. Write one `summary.csv` row per image. For duplicate `label3` or `label5` detections, use the highest detection confidence item in the summary and preserve all rows in `detections.csv`.
6. Draw visualizations on the original image:
   - All OBB detections are drawn.
   - `label3` and `label5` include classification label and confidence.
   - OK boxes are green, NG boxes are red, unclassified boxes are yellow.

## Output Files

Each run writes under `runs/pipeline/<name>/`:

- `detections.csv`: one row per detection.
- `summary.csv`: one row per source image.
- `pipeline_report.txt`: counts and warnings.
- `crops/`: rectified crops for classified detections.
- `visualizations/`: annotated original images.

## CSV Schemas

`detections.csv` fields:

- `image_path`
- `image_name`
- `det_index`
- `det_label`
- `det_conf`
- `x1`, `y1`, `x2`, `y2`, `x3`, `y3`, `x4`, `y4`
- `crop_path`
- `classifier`
- `predicted_label`
- `cls_confidence`
- `prob_NG`
- `prob_OK`

`summary.csv` fields:

- `image_path`
- `image_name`
- `detections`
- `label3_pred`
- `label3_conf`
- `label3_det_conf`
- `label5_pred`
- `label5_conf`
- `label5_det_conf`
- `final_result`
- `warnings`
- `visualization_path`

## Final Result Rule

- If selected `label3` or selected `label5` is `NG`, `final_result=NG`.
- If selected `label3` and selected `label5` are both `OK`, `final_result=OK`.
- If either selected `label3` or selected `label5` is missing, `final_result=UNKNOWN`.

Warnings record missing labels and duplicates:

- `missing label3`
- `missing label5`
- `duplicate label3 count=N`
- `duplicate label5 count=N`

## Implementation Boundaries

- Existing training, evaluate, and predict scripts remain unchanged.
- The pipeline uses trained checkpoints only; it does not train models.
- Classification is initially limited to `label3` and `label5`.
- The pipeline does not depend on YOLO's saved `.txt` output; it uses in-memory Ultralytics result objects.

## Validation

- Unit tests cover summary selection, final result rules, CSV row formatting, and visualization output creation.
- A smoke run should work with local weights and a small image directory when all weight paths are available.
