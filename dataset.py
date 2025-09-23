import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from monai.data import DataLoader, Dataset, CacheDataset, partition_dataset
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from skimage.transform import resize
from PIL import Image





# ======================== SALSA ======================== #
class SalsaHGFAlignedDataset(Dataset):
    """
    读取 SALSA 导出的图像并与 Harvard-GF Test 的标签对齐：
      - 图像目录: image_root/data_xxxx/{rnfl_thickness_map.jpg, slab_image.png, ...}
      - 标签目录: hgf_test_root/data_xxxx.npz  (包含 'tds')

    返回的 sample（根据 modality_type 决定是否含 slab）：
      modality_type='rnflt'       -> {'rnfl': (H,W)uint8, 'label': float32..., 'id': str}
      modality_type='rnflt+slab'  -> {'rnfl': (H,W)uint8, 'slab': (H,W)uint8, 'label': float32..., 'id': str}

    不做任何 resize/normalize/通道处理，交由 transform 自行完成。
    """
    def __init__(
        self,
        image_root: str,
        hgf_test_root: str,
        modality_type: str = 'rnflt',   # 'rnflt' | 'rnflt+slab'
        transform = None
    ):
        assert modality_type in ('rnflt', 'rnflt+slab')
        self.image_root = image_root
        self.hgf_root = hgf_test_root
        self.modality_type = modality_type
        self.transform = transform

        # 用 SALSA 目录名作为样本 id（如 data_3285）
        self.sample_ids = sorted([
            d for d in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, d))
        ])
        assert self.sample_ids, f"No subfolders under {self.image_root}"

        # 基本存在性检查
        missing = []
        for sid in self.sample_ids:
            sdir = os.path.join(self.image_root, sid)
            if not os.path.exists(os.path.join(sdir, 'rnfl_thickness_map.jpg')):
                missing.append(f"{sid}: rnfl_thickness_map.jpg")
            if self.modality_type == 'rnflt+slab':
                if not os.path.exists(os.path.join(sdir, 'slab_image.png')):
                    missing.append(f"{sid}: slab_image.png")
            if not os.path.exists(os.path.join(self.hgf_root, f'{sid}.npz')):
                missing.append(f"{sid}: label npz")
        if missing:
            raise FileNotFoundError("Missing files:\n  - " + "\n  - ".join(missing[:20]) +
                                    (f"\n  ... total {len(missing)} missing" if len(missing) > 20 else ""))

    def __len__(self):
        return len(self.sample_ids)

    def _read_gray_uint8(self, path):
        return np.asarray(Image.open(path).convert('L'))
    
    def _read_rgb(self, path):
        return np.asarray(Image.open(path).convert('RGB'))  # (H, W, 3)

    def _load_label(self, sid):
        npz_path = os.path.join(self.hgf_root, f'{sid}.npz')
        raw = np.load(npz_path, allow_pickle=True)
        if 'tds' not in raw:
            raise KeyError(f"'tds' not in {npz_path}, keys={raw.files}")
        return raw['tds'].astype(np.float32)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sdir = os.path.join(self.image_root, sid)

        rnfl = self._read_rgb(os.path.join(sdir, 'rnfl_thickness_map.jpg'))
        rnfl_chw = np.transpose(rnfl, (2, 0, 1)) 
        sample = {
            'id': sid,
            'image': rnfl_chw,                  # (C, H, W) float32
            'label': self._load_label(sid),    # float32 array
        }
        if self.modality_type == 'rnflt+slab':
            slab = self._read_gray_uint8(os.path.join(sdir, 'slab_image.png'))
            slab3 = np.repeat(slab[None, ...], 3, axis=0).astype(np.float32)     # (3,H,W)
            sample['slab'] = slab3              # (3, H, W) float32

        if  self.transform:
            sample = self.transform(sample)
        
        return sample


def get_dataloader_Salsa(
    image_root,
    hgf_test_root,
    modality_type,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    transform=None,
    ids=None, 
):
    dataset = SalsaHGFAlignedDataset(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        transform=transform
    )

    sampler = None
    if ids is not None:
        # 将 id 映射为索引
        id2idx = {sid: i for i, sid in enumerate(dataset.sample_ids)}
        indices = [id2idx[sid] for sid in ids if sid in id2idx]
        if shuffle:
            sampler = SubsetRandomSampler(indices)
        else:
            sampler = SequentialSampler(indices)

    # 注意：当使用 sampler 时，DataLoader 的 shuffle 必须为 False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )

# ======================== Harvard-GF ======================== #
class HarvardGFDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        modality_type='rnflt', 
        task='tds', 
        resolution=224, 
        depth=1,
        attribute_type=None, 
        transform=None
    ):
        self.data_path = data_path
        self.modality_type = modality_type
        self.task = task
        self.resolution = resolution
        self.depth = depth
        self.attribute_type = attribute_type
        self.transform = transform

        self.files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        assert len(self.files) > 0, f"No .npz files found in {data_path}!"

        self.race_mapping = {'Asian': 0, 'Black or African American': 1, 'White or Caucasian': 2}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.files[idx])
        raw_data = np.load(file_path, allow_pickle=True)

        # --- 数据处理 ---
        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth > 1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)

        elif 'bscan' in self.modality_type:
            oct_img = raw_data['oct_bscans']
            oct_img_array = []
            for img in oct_img:
                oct_img_array.append(resize(img, (self.resolution, self.resolution)))
            oct_img_array = np.stack(oct_img_array, axis=0)
            data_sample = np.stack([oct_img_array]*(1), axis=0).astype(np.float32)  # shape: [1, N, H, W]
            if self.transform:
                data_sample = self.transform(data_sample)
        else:
            raise ValueError(f"Unknown modality_type: {self.modality_type}")

        # --- label处理，只保留tds ---
        if self.task == 'tds':
            label = raw_data['tds'].astype(np.float32)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # --- 属性（可选）---
        attr = None
        if self.attribute_type == 'race' and 'race' in raw_data:
            race = raw_data['race'].item()
            attr = self.race_mapping.get(race, -1)
        elif self.attribute_type == 'gender' and 'male' in raw_data:
            attr = int(raw_data['male'].item())

        # --- 返回格式适配（dict，与HarvardGFDataset保持一致）---
        sample = {
            'image': data_sample,  # [C, H, W]
            'label': label
        }
        if attr is not None:
            sample['attr'] = attr

        if  self.transform:
            sample = self.transform(sample)

        return sample



def get_dataloader_HGF(data_dir, modality_type, task, resolution, depth, batch_size=16, shuffle=True, num_workers=4, transform=None):
    dataset = HarvardGFDataset(data_dir, modality_type, task, resolution, depth, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


# class HarvardGFDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
#         assert len(self.files) > 0, f"No .npz files found in {data_dir}!"

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         npz = np.load(self.files[idx])
#         rnflt = npz["rnflt"].astype(np.float32)
#         tds = npz["tds"].astype(np.float32)
#         rnflt = np.expand_dims(rnflt, axis=0)  # [1, 200, 200]

#         sample = {
#             'image': rnflt,
#             'label': tds
#         }
#         if self.transform:
#             sample = self.transform(sample)
#         return sample





