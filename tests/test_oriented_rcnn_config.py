from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from obb_detection.models.oriented_rcnn.config import (
    find_base_config,
    load_class_names,
    render_oriented_rcnn_config,
    validate_dota_dataset,
)


class OrientedRcnnConfigTests(unittest.TestCase):
    def test_render_uses_requested_data_model_and_schedule(self) -> None:
        config = render_oriented_rcnn_config(
            base_config=Path("/opt/mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py"),
            data_root=Path("/data/terminal_obb"),
            class_names=["label1_thin", "label1_thick", "label2"],
            imgsz=1280,
            epochs=50,
            batch=2,
            workers=4,
            learning_rate=0.005,
            flip_prob=0.75,
        )

        self.assertIn("num_classes=3", config)
        self.assertIn("scale=(1280, 1280)", config)
        self.assertIn("max_epochs = 50", config)
        self.assertIn("batch_size=2", config)
        self.assertIn("ann_file='train/annfiles/'", config)
        self.assertIn("ann_file='test/annfiles/'", config)
        self.assertIn("model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))", config)
        self.assertIn("save_best='dota/mAP'", config)

    def test_find_base_config_and_validate_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py"
            base.parent.mkdir(parents=True)
            base.write_text("# base\n", encoding="utf-8")
            data = root / "data"
            (data / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
            (data / "classes.txt").write_text("label1\nlabel2\n", encoding="utf-8")
            for split in ("train", "test"):
                image_dir = data / split / "images"
                ann_dir = data / split / "annfiles"
                image_dir.mkdir(parents=True)
                ann_dir.mkdir(parents=True)
                (image_dir / "sample.png").touch()
                (ann_dir / "sample.txt").write_text("", encoding="utf-8")

            self.assertEqual(find_base_config(root), base.resolve())
            self.assertEqual(load_class_names(data), ["label1", "label2"])
            self.assertEqual(
                validate_dota_dataset(data),
                {"train": (1, 1), "test": (1, 1)},
            )

    def test_rejects_invalid_numeric_settings(self) -> None:
        with self.assertRaises(ValueError):
            render_oriented_rcnn_config(
                base_config=Path("base.py"),
                data_root=Path("data"),
                class_names=["label1"],
                flip_prob=1.5,
            )

    def test_training_preflight_directory_can_be_reused(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/models/oriented_rcnn/train.py"
        spec = importlib.util.spec_from_file_location("oriented_rcnn_train_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.py").write_text("# generated\n", encoding="utf-8")
            (run_dir / "train_command.txt").write_text("python train.py\n", encoding="utf-8")
            self.assertTrue(module._preflight_only(run_dir))
            (run_dir / "epoch_1.pth").touch()
            self.assertFalse(module._preflight_only(run_dir))


if __name__ == "__main__":
    unittest.main()
