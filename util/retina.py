# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import blobfile as bf
import torch.utils.data as data
import numpy as np
import os
import io
from PIL import Image, ImageOps


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class RetinaDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        if not root:
            raise ValueError("unspecified data directory")
        self.local_images = _list_image_files_recursively(root)
        

    def __getitem__(self, index):
        path = self.local_images[index]
        with bf.BlobFile(path, "rb") as f:
            img = Image.open(f)
            img.load()
        img = img.convert("RGB")
        target = [0]
        if self.transform is not None:
            img = self.transform(img)

        arr = np.array(img)
        arr = arr.astype(np.float32)
        return arr, target

    def __len__(self):
        return len(self.local_images)
