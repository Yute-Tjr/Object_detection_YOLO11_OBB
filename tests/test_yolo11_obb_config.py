import tempfile
import unittest
from pathlib import Path

from yolo11_obb.config import (
    TrainOptions,
    build_train_kwargs,
    load_dataset_config,
    resolve_from_root,
    validate_dataset_layout,
)


def write_file(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class DatasetConfigTests(unittest.TestCase):
    def test_load_dataset_config_resolves_split_paths_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "data.yaml"
            write_file(
                data_yaml,
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "val: images/val",
                        "test: images/test",
                        "names:",
                        "  0: label1",
                        "  1: label2",
                    ]
                )
                + "\n",
            )

            config = load_dataset_config(data_yaml)

            resolved_root = root.resolve()
            self.assertEqual(config.root, resolved_root)
            self.assertEqual(config.splits["train"], resolved_root / "images" / "train")
            self.assertEqual(config.splits["val"], resolved_root / "images" / "val")
            self.assertEqual(config.splits["test"], resolved_root / "images" / "test")
            self.assertEqual(config.names, {0: "label1", 1: "label2"})

    def test_load_dataset_config_resolves_project_relative_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            dataset_root = project / "datasets" / "sample"
            data_yaml = dataset_root / "data.yaml"
            write_file(
                data_yaml,
                "\n".join(
                    [
                        "path: datasets/sample",
                        "train: images/train",
                        "val: images/test",
                        "test: images/test",
                        "names:",
                        "  0: label1",
                    ]
                )
                + "\n",
            )

            config = load_dataset_config(data_yaml)

            self.assertEqual(config.root, dataset_root.resolve())
            self.assertEqual(config.splits["train"], dataset_root.resolve() / "images" / "train")
            self.assertEqual(config.splits["val"], dataset_root.resolve() / "images" / "test")

    def test_validate_dataset_layout_requires_matching_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                write_file(root / "images" / split / "sample.bmp")
                write_file(root / "labels" / split / "sample.txt", "0 0 0 1 0 1 1 0 1\n")

            report = validate_dataset_layout(root)

            self.assertEqual(report["train"].image_count, 1)
            self.assertEqual(report["val"].label_count, 1)
            self.assertEqual(report["test"].missing_labels, [])
            self.assertEqual(report["test"].missing_images, [])

            (root / "labels" / "val" / "sample.txt").unlink()

            with self.assertRaises(ValueError) as ctx:
                validate_dataset_layout(root)

            self.assertIn("val", str(ctx.exception))
            self.assertIn("missing labels", str(ctx.exception))

    def test_validate_dataset_layout_uses_configured_split_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "data.yaml"
            write_file(root / "images" / "train" / "train_sample.bmp")
            write_file(root / "labels" / "train" / "train_sample.txt", "0 0 0 1 0 1 1 0 1\n")
            write_file(root / "images" / "test" / "test_sample.bmp")
            write_file(root / "labels" / "test" / "test_sample.txt", "0 0 0 1 0 1 1 0 1\n")
            write_file(
                data_yaml,
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "val: images/test",
                        "test: images/test",
                        "names:",
                        "  0: label1",
                    ]
                )
                + "\n",
            )

            dataset = load_dataset_config(data_yaml)
            report = validate_dataset_layout(dataset)

            self.assertEqual(report["train"].image_count, 1)
            self.assertEqual(report["val"].image_count, 1)
            self.assertEqual(report["test"].label_count, 1)

    def test_build_train_kwargs_uses_yolo11_obb_defaults(self) -> None:
        options = TrainOptions(
            data=Path("dataset/data.yaml"),
            model="yolo11n-obb.pt",
            epochs=150,
            imgsz=1024,
            batch=8,
            device="cpu",
            project=Path("runs/obb"),
            name="terminal_obb_yolo11n",
        )

        kwargs = build_train_kwargs(options)

        self.assertEqual(kwargs["data"], "dataset/data.yaml")
        self.assertEqual(kwargs["epochs"], 150)
        self.assertEqual(kwargs["imgsz"], 1024)
        self.assertEqual(kwargs["batch"], 8)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["project"], "runs/obb")
        self.assertEqual(kwargs["name"], "terminal_obb_yolo11n")
        self.assertEqual(kwargs["task"], "obb")
        self.assertEqual(kwargs["val"], True)

    def test_build_train_kwargs_can_disable_training_validation(self) -> None:
        options = TrainOptions(
            data=Path("dataset/data.yaml"),
            validate=False,
        )

        kwargs = build_train_kwargs(options)

        self.assertEqual(kwargs["val"], False)

    def test_resolve_from_root_anchors_relative_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            resolved = resolve_from_root(Path("runs/obb"), root)

            self.assertEqual(resolved, root.resolve() / "runs" / "obb")


if __name__ == "__main__":
    unittest.main()
