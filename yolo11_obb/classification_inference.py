from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from .config import IMAGE_EXTENSIONS


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform=None) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


def build_resnet18_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def classification_transform(imgsz: int):
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def normalize_device_name(requested: str | None) -> str | None:
    if requested is None:
        return None
    text = str(requested).strip()
    if text.isdigit():
        return f"cuda:{text}"
    return text


def select_device(requested: str | None = None) -> torch.device:
    normalized = normalize_device_name(requested)
    if normalized:
        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested but MPS is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_resnet18_checkpoint(weights_path: Path, device: torch.device) -> tuple[nn.Module, List[str], dict]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    classes = [str(class_name) for class_name in checkpoint["classes"]]
    model = build_resnet18_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, classes, checkpoint


def collect_image_paths(source: Path) -> List[Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"source does not exist: {source}")
    image_paths = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"no images found in source: {source}")
    return image_paths


def prediction_rows(
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    probabilities: Iterable[Sequence[float]],
) -> List[dict[str, str]]:
    rows: List[dict[str, str]] = []
    for image_path, probability_row in zip(image_paths, probabilities):
        values = [float(value) for value in probability_row]
        predicted_index = max(range(len(values)), key=lambda index: values[index])
        row = {
            "image_path": str(image_path),
            "predicted_label": class_names[predicted_index],
            "confidence": f"{values[predicted_index]:.6f}",
        }
        for class_name, probability in zip(class_names, values):
            row[f"prob_{class_name}"] = f"{probability:.6f}"
        rows.append(row)
    return rows
