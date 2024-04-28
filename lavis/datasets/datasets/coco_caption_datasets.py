"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset, ProteinDataset, ProteinEvalDataset

COCOCapDataset = CaptionDataset
ProteinCapDataset = ProteinDataset


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


# custom protein dataset
class ProteinFuncEvalDataset(ProteinEvalDataset):
    def __init__(self, text_processor, ann_paths):
        super().__init__(text_processor, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        name = ann['image']
        #image_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2/{}.pt'.format(name))['representations'][33]
        image_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2_3b/{}.pt'.format(name))['representations'][36]
        image_emb = F.pad(image_emb.t(), (0, 1024 - len(image_emb))).t()

        #caption = self.text_processor(ann['caption'])
        caption = ann['caption']
        name = ann['image']
        if ann['prompt'] == 'none':
            prompt = ''
        else:
            prompt = ann['prompt']

        return {
            "image": image_emb,
            "text_input": caption,
            "name": name,
            'prompt': prompt,
            "image_id": ann["image_id"],
            # "image_id": self.img_ids[ann["image_id"]],
        }
