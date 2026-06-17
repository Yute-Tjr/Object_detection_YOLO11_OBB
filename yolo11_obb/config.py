from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union

"""

dataclass 装饰器：直接将数据转为该对象的属性

"""

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetConfig:
    """
    保存data.yaml解析之后的结果
    数据集的根目录
    train/val/test 路径、类别编号到类别名的映射。
    """
    data_yaml: Path
    root: Path
    splits: Dict[str, Path]
    names: Dict[int, str]


@dataclass(frozen=True)
class SplitReport:
    """
    记录每个数据划分的检查结果。
    例如训练集有多少图片、多少标签、有没有图片缺标签、有没有标签缺图片
    """
    split: str
    image_count: int
    label_count: int
    missing_labels: List[str]
    missing_images: List[str]


@dataclass(frozen=True)
class TrainOptions:
    """
    训练需要的参数
    """
    data: Path
    model: str = "yolo11n-obb.pt"
    epochs: int = 150
    imgsz: int = 1024
    batch: int = 8
    device: Optional[str] = None
    project: Path = Path("runs/obb")
    name: str = "terminal_obb_yolo11n"
    patience: int = 30
    workers: int = 4
    seed: int = 42
    exist_ok: bool = False
    validate: bool = True


@dataclass(frozen=True)
class EvalOptions:
    """
    评估需要的参数
    """
    data: Path
    model: Path
    split: str = "test"
    imgsz: int = 1024
    batch: int = 8
    device: Optional[str] = None
    project: Path = Path("runs/obb")
    name: str = "terminal_obb_eval"


@dataclass(frozen=True)
class PredictOptions:
    """
    预测需要的参数
    """
    model: Path
    source: Path
    imgsz: int = 1024
    conf: float = 0.25
    device: Optional[str] = None
    project: Path = Path("runs/obb")
    name: str = "terminal_obb_predict"
    save_txt: bool = True
    save_conf: bool = True


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_simple_yaml(path: Path) -> Dict[str, object]:
    """
    解析data.yaml
    Parse the simple data.yaml format used by Ultralytics datasets.
    This intentionally supports only the subset we need: scalar keys and
    integer-indexed nested maps such as `names`.
    """
    parsed: Dict[str, object] = {}
    current_map: Optional[str] = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        if current_map and indent > 0:
            if ":" not in line:
                raise ValueError(f"Invalid nested YAML line in {path}: {raw}")
            key, value = line.split(":", 1)
            nested = parsed.setdefault(current_map, {})
            if not isinstance(nested, dict):
                raise ValueError(f"YAML key is not a mapping: {current_map}")
            nested[int(_strip_quotes(key))] = _strip_quotes(value)
            continue

        current_map = None
        if ":" not in line:
            raise ValueError(f"Invalid YAML line in {path}: {raw}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parsed[key] = _strip_quotes(value)
        else:
            parsed[key] = {}
            current_map = key

    return parsed


def _resolve_split(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset split path must be a non-empty string, got {value!r}")
    split_path = Path(value)
    return split_path if split_path.is_absolute() else root / split_path


def _resolve_dataset_root(yaml_path: Path, root_value: str) -> Path:
    root = Path(root_value).expanduser()
    if root.is_absolute():
        return root.resolve()

    candidates = [yaml_path.parent / root]
    candidates.extend(parent / root for parent in yaml_path.parents)
    candidates.append(Path.cwd() / root)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_from_root(path: Union[str, Path], root: Union[str, Path]) -> Path:
    """
    把相对路径固定解析到项目根目录。防止出现嵌套
    :param path:
    :param root:
    :return:
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(root).expanduser().resolve() / candidate).resolve()


def load_dataset_config(data_yaml: Union[str, Path]) -> DatasetConfig:
    """
    读取并解析 data.yaml，得到 DatasetConfig。训练前会用它确认数据路径和类别名。
    :param data_yaml:
    :return:
    """
    yaml_path = Path(data_yaml).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {yaml_path}")

    raw = _parse_simple_yaml(yaml_path)
    root_value = raw.get("path", str(yaml_path.parent))
    if not isinstance(root_value, str):
        raise ValueError(f"`path` in {yaml_path} must be a string")

    root = _resolve_dataset_root(yaml_path, root_value)

    splits = {
        "train": _resolve_split(root, raw.get("train")),
        "val": _resolve_split(root, raw.get("val")),
        "test": _resolve_split(root, raw.get("test")),
    }

    names_raw = raw.get("names")
    if not isinstance(names_raw, dict) or not names_raw:
        raise ValueError(f"`names` mapping is required in {yaml_path}")
    names = {int(idx): str(name) for idx, name in names_raw.items()}

    return DatasetConfig(
        data_yaml=yaml_path.resolve(),
        root=root,
        splits=splits,
        names=names,
    )


def _image_stems(image_dir: Path) -> Iterable[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split directory not found: {image_dir}")
    return sorted(
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_stems(label_dir: Path) -> Iterable[str]:
    if not label_dir.exists():
        raise FileNotFoundError(f"Label split directory not found: {label_dir}")
    return sorted(path.stem for path in label_dir.glob("*.txt"))


def _label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def validate_dataset_layout(root: Union[str, Path, DatasetConfig]) -> Dict[str, SplitReport]:
    """
    检查数据集结构是否正确。它会确认：

    images/train 和 labels/train 是否一一对应
    images/val 和 labels/val 是否一一对应
    images/test 和 labels/test 是否一一对应

    这是为了避免训练跑到一半才发现某张图没有标签。
    :param root:
    :return:
    """
    if isinstance(root, DatasetConfig):
        split_items = [(split, path.resolve()) for split, path in root.splits.items()]
    else:
        dataset_root = Path(root).expanduser().resolve()
        split_items = [
            ("train", dataset_root / "images" / "train"),
            ("val", dataset_root / "images" / "val"),
            ("test", dataset_root / "images" / "test"),
        ]
    reports: Dict[str, SplitReport] = {}
    errors: List[str] = []

    for split, image_dir in split_items:
        label_dir = _label_dir_for_image_dir(image_dir)
        images = set(_image_stems(image_dir))
        labels = set(_label_stems(label_dir))
        missing_labels = sorted(images - labels)
        missing_images = sorted(labels - images)
        reports[split] = SplitReport(
            split=split,
            image_count=len(images),
            label_count=len(labels),
            missing_labels=missing_labels,
            missing_images=missing_images,
        )
        if missing_labels:
            errors.append(f"{split}: missing labels for {missing_labels[:5]}")
        if missing_images:
            errors.append(f"{split}: missing images for {missing_images[:5]}")

    if errors:
        raise ValueError("; ".join(errors))
    return reports


def build_train_kwargs(options: TrainOptions) -> Dict[str, object]:
    """
    将参数类转换为Yolo需要的字典参数
    :param options:
    :return:
    """
    kwargs: Dict[str, object] = {
        "data": str(options.data),
        "epochs": options.epochs,
        "imgsz": options.imgsz,
        "batch": options.batch,
        "project": str(options.project),
        "name": options.name,
        "patience": options.patience,
        "workers": options.workers,
        "seed": options.seed,
        "exist_ok": options.exist_ok,
        "task": "obb",
        "val": options.validate,
    }
    if options.device:
        kwargs["device"] = options.device
    return kwargs


def build_eval_kwargs(options: EvalOptions) -> Dict[str, object]:
    """
    将参数类转换为Yolo需要的字典参数
    :param options:
    :return:
    """
    kwargs: Dict[str, object] = {
        "data": str(options.data),
        "split": options.split,
        "imgsz": options.imgsz,
        "batch": options.batch,
        "project": str(options.project),
        "name": options.name,
        "task": "obb",
    }
    if options.device:
        kwargs["device"] = options.device
    return kwargs


def build_predict_kwargs(options: PredictOptions) -> Dict[str, object]:
    """
    将参数类转换为Yolo需要的字典参数
    :param options:
    :return:
    """
    kwargs: Dict[str, object] = {
        "source": str(options.source),
        "imgsz": options.imgsz,
        "conf": options.conf,
        "project": str(options.project),
        "name": options.name,
        "save": True,
        "save_txt": options.save_txt,
        "save_conf": options.save_conf,
        "task": "obb",
    }
    if options.device:
        kwargs["device"] = options.device
    return kwargs


def format_layout_report(report: Mapping[str, SplitReport]) -> str:
    lines = []
    ordered_splits = [split for split in ("train", "val", "test") if split in report]
    ordered_splits.extend(split for split in report if split not in ordered_splits)
    for split in ordered_splits:
        item = report[split]
        lines.append(
            f"{split}: {item.image_count} images, {item.label_count} labels"
        )
    return "\n".join(lines)
