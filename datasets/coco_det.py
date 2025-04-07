# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union, Dict, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from pycocotools import mask as coco_mask

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

# Standard COCO detection class mapping (adjust if necessary)
# Maps original COCO IDs (1-90) to contiguous IDs (0-79)
# Source: https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
COCO_DET_CLASS_MAPPING = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28,
    34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37,
    43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46,
    53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55,
    62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64,
    75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


class COCODetection(LightningDataModule):
    """
    LightningDataModule for COCO Object Detection.
    Loads images and annotations from COCO dataset structure.
    
    The dataset provides:
    - Images
    - Object detection annotations as masks:
      * Converting bounding boxes to rectangular masks
      * Class labels
      * Crowd flags
    """
    def __init__(
        self,
        path: str,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640), # Note: img_size is for transforms
        num_classes: int = 80, # COCO detection has 80 classes
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True, # Filter images with no annotations
    ) -> None:
        """
        Args:
            path: Path to the COCO dataset directory. Should contain train2017.zip,
                  val2017.zip, and annotations_trainval2017.zip.
            num_workers: Number of workers for DataLoader.
            batch_size: Batch size for DataLoader.
            img_size: Target image size (height, width) for transforms and final output.
            num_classes: Number of object classes (default 80 for COCO).
            color_jitter_enabled: Whether to apply color jitter augmentation.
            scale_range: Range for random scaling augmentation.
            check_empty_targets: If True, skips images with no valid annotations.
        """
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"]) # Assuming OmegaConf is used

        # Transforms might need adjustment for bounding boxes
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        bbox_by_id: dict[int, list[float]] = None,  # Add bounding box parameter
        **kwargs
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            if cls_id not in COCO_DET_CLASS_MAPPING:
                continue

            x, y, w, h = bbox_by_id[label_id]
            # Convert to integer coordinates
            x1, y1 = round(x), round(y)
            x2, y2 = round(x + w), round(y + h)
            
            # Create empty mask of correct size
            mask = torch.zeros((height, width), dtype=torch.bool)
            # Set the rectangular region to True
            mask[y1:y2, x1:x2] = True
            masks.append(tv_tensors.Mask(mask))

            labels.append(COCO_DET_CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> None:
        """
        Sets up the dataset for training and validation.
        """
        # Adjusted for COCO Detection structure
        dataset_kwargs = {
            "img_suffix": ".jpg",
            # Use the static method defined above to parse annotations into masks
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
            # Rely solely on the JSON for target information (bboxes, labels)
            "only_annotations_json": True, # Rely on JSON for detection targets
            "target_suffix": None, # Not using target images
        }
        self.train_dataset = Dataset(
            transforms=self.transforms, # Pass transforms
            img_folder_path_in_zip=Path("./train2017"),
            # No target_folder_path_in_zip needed for detection via JSON
            annotations_json_path_in_zip=Path("./annotations/instances_train2017.json"), # Detection annotations
            # No target_zip_path_in_zip needed
            # target_zip_path should point to the zip containing annotations
            target_zip_path=Path(self.path, "annotations_trainval2017.zip"), # Assuming annotations are zipped
            zip_path=Path(self.path, "train2017.zip"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            # No transforms for validation by default, can be added if needed
            img_folder_path_in_zip=Path("./val2017"),
            # No target_folder_path_in_zip needed
            annotations_json_path_in_zip=Path("./annotations/instances_val2017.json"), # Detection annotations
            # No target_zip_path_in_zip needed
            target_zip_path=Path(self.path, "annotations_trainval2017.zip"), # Assuming annotations are zipped
            zip_path=Path(self.path, "val2017.zip"),
            **dataset_kwargs,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate, # From LightningDataModule base
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset, # Original dataset
            shuffle=False,
            collate_fn=self.eval_collate, # From LightningDataModule base
            **self.dataloader_kwargs,
        )

    # test_dataloader can be added similarly if needed


