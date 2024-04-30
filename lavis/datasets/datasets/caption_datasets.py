"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import torch
import pandas as pd
import random
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


# custom protein dataset
class ProteinDataset(__DisplMixin):
    def __init__(self, text_processor, ann_paths):
        """
        vis_root (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.annotation = []
        print(ann_paths)
        if 'pkl' in ann_paths[0]:
            df_ann = pd.read_pickle(ann_paths[0])
        elif 'csv' in ann_paths[0]:
            df_ann = pd.read_csv(ann_paths[0], sep='|')
        #for i, j, l in zip(df_ann['id'], df_ann['name'], df_ann['function']):
        #    self.annotation.append({'image_id': i, 'image': j, 'caption': l})
        
        if 'prompt' in list(df_ann.columns):
            print("Use prompt!")
            for i, j, l, m in zip(df_ann['id'], df_ann['name'], df_ann['function'], df_ann['prompt']):
                self.annotation.append({'image_id': i, 'image': j, 'caption': l, 'prompt': m})
        else:
            print("Not use prompt!")
            for i, j, l in zip(df_ann['id'], df_ann['name'], df_ann['function']):
                self.annotation.append({'image_id': i, 'image': j, 'caption': l, 'prompt': ''})

        del df_ann

        self.text_processor = text_processor

        self._add_instance_ids()

        #self.img_ids = {}
        #n = 0
        #for ann in self.annotation:
        #    img_id = ann["image_id"]
        #    if img_id not in self.img_ids.keys():
        #        self.img_ids[img_id] = n
        #        n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        name = ann['image']
        #image_emb = torch.load('data/emb_esm2_650m/{}.pt'.format(name))['representations'][33]
        image_emb = torch.load('data/emb_esm2_3b/{}.pt'.format(name))['representations'][36]
        image_emb = F.pad(image_emb.t(), (0, 1024 - len(image_emb))).t()
        
        caption = self.text_processor(ann['caption'])
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
            "image_id": ann["image_id"]
            #"image_id": self.img_ids[ann["image_id"]],
        }

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, text_processor):
        self.text_processor = text_processor


class ProteinEvalDataset(__DisplMixin):
    def __init__(self, text_processor, ann_paths):
        """
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.annotation = []
        #df_ann = pd.read_csv(ann_paths[0], sep='|', usecols=['id', 'name', 'protein', 'function'])
        if 'pkl' in ann_paths[0]:
            df_ann = pd.read_pickle(ann_paths[0])
        elif 'csv' in ann_paths[0]:
            df_ann = pd.read_csv(ann_paths[0], sep='|')
        if 'prompt' in list(df_ann.columns):
            print("Use prompt!")
            for i, j, l, m in zip(df_ann['id'], df_ann['name'], df_ann['function'], df_ann['prompt']):
                self.annotation.append({'image_id': i, 'image': j, 'caption': l, 'prompt': m})
        else:
            print("Not use prompt!")
            for i, j, l in zip(df_ann['id'], df_ann['name'], df_ann['function']):
                self.annotation.append({'image_id': i, 'image': j, 'caption': l, 'prompt': ''})

        self.text_processor = text_processor

        self._add_instance_ids()

        #self.img_ids = {}
        #n = 0
        #for ann in self.annotation:
        #    img_id = ann["image_id"]
        #    if img_id not in self.img_ids.keys():
        #        self.img_ids[img_id] = n
        #        n += 1
    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        name = ann['image']
        #image_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2/{}.pt'.format(name))['representations'][33]
        image_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2_3b/{}.pt'.format(name))['representations'][36]
        image_emb = F.pad(image_emb.t(), (0, 1024 - len(image_emb))).t()

        caption = self.text_processor(ann['caption'])
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
            #"image_id": self.img_ids[ann["image_id"]],
        }

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, text_processor):
        self.text_processor = text_processor

