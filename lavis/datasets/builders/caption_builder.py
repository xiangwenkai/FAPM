"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, ProteinDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
    ProteinCapDataset,
    ProteinFuncEvalDataset,
)

from lavis.common.registry import registry



@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


# custom dataset
@registry.register_builder("protein_function")
class ProteinCapBuilder(ProteinDatasetBuilder):
    train_dataset_cls = ProteinCapDataset
    eval_dataset_cls = ProteinFuncEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/protein/defaults_cap.yaml",
    }


# custom dataset
@registry.register_builder("GO_protein_function")
class ProteinCapBuilder(ProteinDatasetBuilder):
    train_dataset_cls = ProteinCapDataset
    eval_dataset_cls = ProteinFuncEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/protein/GO_defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }

