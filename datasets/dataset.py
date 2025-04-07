# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import re
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading images and corresponding targets (masks/annotations)
    from zip archives. Supports loading from separate zip files for images, semantic
    targets, and instance targets, potentially nested within other zips. Can handle
    annotations provided via a JSON file in COCO format or directly from target images.

    Handles efficient loading in multi-worker scenarios by maintaining separate zip file
    handles per worker.
    """
    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
    ):
        """
        Initializes the Dataset object.

        Args:
            zip_path: Path to the main zip archive containing images.
            img_suffix: Suffix of the image files (e.g., ".jpg", ".png").
            target_parser: A callable function that parses target data (semantic/instance masks
                           or annotations) and returns masks, labels, and is_crowd flags.
            check_empty_targets: If True, filter out images whose corresponding target masks
                                 are entirely empty (single value).
            transforms: Optional torchvision transforms to apply to image and target pairs.
            only_annotations_json: If True, only use annotations from the JSON file for target
                                   information, ignoring target images.
            target_suffix: Suffix of the target mask files (e.g., ".png"). Required if
                           `only_annotations_json` is False.
            stuff_classes: Optional list of class IDs considered "stuff" categories. Used by
                           the `target_parser`.
            img_stem_suffix: Suffix to remove from image filename stems before matching with
                             target filenames (e.g., "_leftImg8bit").
            target_stem_suffix: Suffix to add to image filename stems (after removing
                                `img_stem_suffix`) to form target filenames.
            target_zip_path: Optional path to a separate zip archive for semantic target masks.
                             If None, targets are assumed to be in the `zip_path` archive.
            target_zip_path_in_zip: Optional path *within* `target_zip_path` (or `zip_path` if
                                   `target_zip_path` is None) pointing to a nested zip archive
                                   containing the target masks.
            target_instance_zip_path: Optional path to a separate zip archive for instance
                                      target masks.
            img_folder_path_in_zip: Path within the image zip archive where images are located.
            target_folder_path_in_zip: Path within the target zip archive where semantic masks
                                       are located.
            target_instance_folder_path_in_zip: Path within the instance target zip archive
                                                where instance masks are located.
            annotations_json_path_in_zip: Optional path *within* the target zip archive
                                          (or `zip_path` if `target_zip_path` is None) to a
                                          COCO-style annotations JSON file.
        """
        self.zip_path = zip_path
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_zip_path = target_zip_path
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.target_instance_zip_path = target_instance_zip_path
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_instance_folder_path_in_zip = target_instance_folder_path_in_zip

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None
        img_zip, target_zip, target_instance_zip = self._load_zips()

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.bbox_by_id = {}  # Initialize bbox dictionary

        if annotations_json_path_in_zip is not None:
            with zipfile.ZipFile(target_zip_path or zip_path) as outer_target_zip:
                with outer_target_zip.open(
                    str(annotations_json_path_in_zip), "r"
                ) as file:
                    annotation_data = json.load(file)

            image_id_to_file_name = {
                image["id"]: image["file_name"] for image in annotation_data["images"]
            }

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}
                    
                    if img_filename not in self.bbox_by_id:
                        self.bbox_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )
                    
                    # Extract and store bounding box if available
                    if "bbox" in annotation:
                        self.bbox_by_id[img_filename][annotation["id"]] = annotation["bbox"]

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        target_zip_filenames = target_zip.namelist()

        for img_info in sorted(img_zip.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                continue

            img_path = Path(img_info.filename)
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_path_in_zip)
                target_parent = target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)

                target_filename = target_parent / f"{target_stem}{target_suffix}"
                target_filename = str(target_filename)

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue

                if not self.labels_by_id[img_path.name]:
                    continue
            else:
                if target_filename not in target_zip_filenames:
                    continue

                if check_empty_targets:
                    with target_zip.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            if target_instance_zip is not None:
                target_instance_filename = str(
                    target_instance_folder_path_in_zip / (target_stem + target_suffix)
                )

                if check_empty_targets:
                    with target_instance_zip.open(
                        target_instance_filename
                    ) as target_instance:
                        extrema = Image.open(target_instance).getextrema()
                        if all(min_val == max_val for min_val, max_val in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(
                                    Image.open(target_zip.open(target_filename))
                                ),
                                target_instance=tv_tensors.Mask(
                                    Image.open(target_instance)
                                ),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            self.imgs.append(str(img_path))

            if not only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_zip is not None:
                self.targets_instance.append(target_instance_filename)

    def __getitem__(self, index: int):
        """
        Retrieves the image and its corresponding target information at the given index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - img (tv_tensors.Image): The loaded image tensor.
            - target (dict): A dictionary containing target information:
                - "masks" (tv_tensors.Mask): Tensor of binary masks for each object instance.
                - "labels" (torch.Tensor): Tensor of labels for each instance.
                - "is_crowd" (torch.Tensor): Tensor indicating if each instance is a crowd region.
        """
        img_zip, target_zip, target_instance_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_zip.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_zip.open(
                self.targets_instance[index]
            ) as target_instance:
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance), dtype=torch.long
                )

        # The target_parser in COCODetection returns masks, labels, is_crowd
        parser_result = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(Path(self.imgs[index]).name, {}),
            labels_by_id=self.labels_by_id.get(Path(self.imgs[index]).name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(Path(self.imgs[index]).name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
            bbox_by_id=self.bbox_by_id.get(Path(self.imgs[index]).name, {}) if hasattr(self, 'bbox_by_id') else None,
        )
        
        # All target_parser implementations return masks, labels, is_crowd
        masks, labels, is_crowd = parser_result

        # Check if the target parser returned empty lists (no valid annotations)
        if not masks: # Check if the masks list is empty
            # Create empty tensors with appropriate shapes and types
            h, w = img.shape[-2:]
            target = {
                "masks": tv_tensors.Mask(torch.empty((0, h, w), dtype=torch.bool)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "is_crowd": torch.empty((0,), dtype=torch.bool),
            }
        else:
            # If annotations exist, stack and convert to tensors
            target = {
                "masks": tv_tensors.Mask(torch.stack(masks)),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _load_zips(
        self,
    ) -> Tuple[zipfile.ZipFile, zipfile.ZipFile, Optional[zipfile.ZipFile]]:
        """
        Loads or retrieves cached zip file handles for the current worker.

        This method ensures that each data loading worker in a DataLoader has its own
        set of zip file handles to avoid potential issues with concurrent access.
        Handles loading of primary image zip, target zip (potentially nested), and
        instance target zip.

        Returns:
            A tuple containing the zip file handles for:
            - Image archive
            - Target archive (semantic masks or annotations)
            - Instance target archive (or None if not applicable)
        """
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)
        if worker not in self.target_zip:
            if self.target_zip_path:
                self.target_zip[worker] = zipfile.ZipFile(self.target_zip_path)
                if self.target_zip_path_in_zip:
                    with self.target_zip[worker].open(
                        str(self.target_zip_path_in_zip)
                    ) as target_zip_stream:
                        nested_zip_data = BytesIO(target_zip_stream.read())
                    self.target_zip[worker].close()
                    self.target_zip[worker] = zipfile.ZipFile(nested_zip_data)
            else:
                self.target_zip[worker] = self.zip[worker]
        if (
            self.target_instance_zip_path is not None
            and worker not in self.target_instance_zip
        ):
            self.target_instance_zip[worker] = zipfile.ZipFile(
                self.target_instance_zip_path
            )

        return (
            self.zip[worker],
            self.target_zip[worker],
            self.target_instance_zip[worker] if self.target_instance_zip_path else None,
        )

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        """
        Generates a sort key for zip file members based on the first number found
        in the filename, falling back to the filename itself. Used for consistent
        ordering of files read from the zip archive.

        Args:
            m: A zipfile.ZipInfo object representing a member of the zip archive.

        Returns:
            A tuple (numeric_part, filename) for sorting.
        """
        match = re.search(r"\d+", m.filename)

        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def valid_member(
        img_info: zipfile.ZipInfo,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        """
        Checks if a zip archive member represents a valid image file based on its
        path, suffix, and type (not a directory).

        Args:
            img_info: The zipfile.ZipInfo object for the archive member.
            img_folder_path_in_zip: The expected parent folder within the zip.
            img_stem_suffix: The expected stem suffix for the image filename.
            img_suffix: The expected file extension suffix for the image.

        Returns:
            True if the member is a valid image file, False otherwise.
        """
        return (
            Path(img_info.filename).is_relative_to(img_folder_path_in_zip)
            and img_info.filename.endswith(img_stem_suffix + img_suffix)
            and not img_info.is_dir()
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.imgs)

    def close(self):
        """Closes all open zip file handles across all workers."""
        if self.zip is not None:
            for item in self.zip.values():
                item.close()
            self.zip = None

        if self.target_zip is not None:
            for item in self.target_zip.values():
                item.close()
            self.target_zip = None

        if self.target_instance_zip is not None:
            for item in self.target_instance_zip.values():
                item.close()
            self.target_instance_zip = None

    def __del__(self):
        """Ensures zip files are closed when the Dataset object is deleted."""
        self.close()

    def __getstate__(self):
        """
        Prepares the dataset state for pickling (serialization).

        Removes the open zip file handles from the state, as they cannot be pickled.
        The `_load_zips` method will re-open them in the new process/worker.

        Returns:
            A dictionary representing the object's state without the zip handles.
        """
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state