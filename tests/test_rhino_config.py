import unittest
from pathlib import Path

from yolo11_obb.rhino_config import render_rhino_config


class RhinoConfigTests(unittest.TestCase):
    def test_kld_config_uses_requested_dataset_and_training_settings(self) -> None:
        config = render_rhino_config(
            base_config=Path("/opt/RHINO/configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py"),
            data_root=Path("/data/rhino_obb"),
            class_names=["label1_thin", "label1_thick"],
            variant="kld",
            imgsz=1280,
            epochs=50,
            batch=2,
            workers=4,
        )

        self.assertIn("num_classes=2", config)
        self.assertIn("scale=(1280, 1280)", config)
        self.assertIn("max_epochs = 50", config)
        self.assertIn("loss_type='kld'", config)
        self.assertIn("ann_file='train/annfiles/'", config)
        self.assertIn("test_evaluator = dict(_delete_=True", config)

    def test_riou_config_replaces_kld_matching_and_regression(self) -> None:
        config = render_rhino_config(
            base_config=Path("/opt/RHINO/configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py"),
            data_root=Path("/data/rhino_obb"),
            class_names=["label1_thin", "label1_thick"],
            variant="riou",
            imgsz=1280,
            epochs=50,
            batch=2,
            workers=4,
        )

        self.assertIn("type='RotatedIoULoss'", config)
        self.assertIn("type='RotatedIoUCost'", config)
        self.assertNotIn("loss_type='kld'", config)


if __name__ == "__main__":
    unittest.main()
