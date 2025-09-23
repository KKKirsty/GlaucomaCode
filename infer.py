#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Optional
from torch import nn
import timm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import HarvardGFDataset
from utils import get_vit_transform, get_cnn_transform, get_albumentations_transform, evaluate_and_generate_predictions, get_imagenet_transform
from dino_model import DinoV3Backbone52


# ---------- 工具：列出/挑选 ID ----------
def _list_all_ids(hgf_dir: str) -> List[str]:
    ids = sorted([Path(f).stem for f in os.listdir(hgf_dir) if f.endswith(".npz")])
    if not ids:
        raise RuntimeError(f"No .npz files found in {hgf_dir}")
    return ids

def _pick_ids(hgf_dir: str, mode: str, n: int, ids: Optional[List[str]]) -> List[str]:
    all_ids = _list_all_ids(hgf_dir)
    if mode == "first_n":
        assert n > 0, "--n must be > 0 for mode=first_n"
        return all_ids[:n]
    elif mode == "ids":
        assert ids and len(ids) > 0, "mode=ids requires --ids/--ids_file"
        aset = set(all_ids)
        return [i for i in ids if i in aset]     # 按传入顺序
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ---------- 构造 tar 表（来自 .npz 的 tds）----------
def _build_tar_df_from_npz(hgf_dir: str, picked_ids: List[str], lat_fill: str = "Unknown") -> pd.DataFrame:
    vecs = []
    for sid in picked_ids:
        npz_path = os.path.join(hgf_dir, f"{sid}.npz")
        raw = np.load(npz_path, allow_pickle=True)
        if "tds" not in raw.files:
            raise KeyError(f"'tds' not in {npz_path}, keys={raw.files}")
        vecs.append(raw["tds"].astype(np.float32))
    mat = np.stack(vecs, axis=0)  # (N, K)
    cols = [f"tar_{i}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "test_id", picked_ids)
    df.insert(1, "test_lat", [lat_fill] * len(picked_ids))
    return df

# ---------- 顺序 sampler：严格按 indices ----------
class _SeqOnIndices(torch.utils.data.Sampler):
    def __init__(self, indices: List[int]):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

# ---------- DataLoader：用现有 HarvardGFDataset ----------
def _get_hgf_loader_subset(
    hgf_dir: str,
    modality_type: str,
    task: str,
    resolution: int,
    depth: int,
    transform,
    picked_ids: List[str],
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    ds = HarvardGFDataset(
        data_path=hgf_dir,
        modality_type=modality_type,
        task=task,
        resolution=resolution,
        depth=depth,
        transform=transform,
    )
    id2idx = {Path(f).stem: i for i, f in enumerate(ds.files)}
    indices = [id2idx[s] for s in picked_ids if s in id2idx]
    sampler = _SeqOnIndices(indices)
    return DataLoader(
        ds,
        batch_size=batch_size,   # 建议=1，兼容官方评估里的 pred[0]
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser("Harvard-GF inference")
    # model
    ap.add_argument('--backbone', type=str, default='dinov3',
                    choices=['vit', 'dinov3'], help='Backbone: vit or dinov3')
    ap.add_argument('--dinov3_model', type=str,
                    default='/mnt/sda/sijiali/GlaucomaCode/pretrained_weight/dinov3-vitb16-pretrain-lvd1689m',
                    help='HuggingFace DINOv3 model name or path')
    # 数据集
    ap.add_argument("--hgf_dir", default='/mnt/sda/sijiali/DataSet/Harvard-GF/Dataset/Test', help="Harvard-GF .npz 文件目录")
    ap.add_argument("--modality_type", choices=["rnflt"], default="rnflt")
    ap.add_argument("--task", choices=["tds"], default="tds")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--depth", type=int, default=3)

    # 子集
    ap.add_argument("--mode", choices=["first_n","ids"], default="first_n")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--ids_file", type=str, default="")

    # 推理
    ap.add_argument("--weight_path", default='/mnt/sda/sijiali/GlaucomaCode/Results_HGF/dinov3/dinov3_imagenet_lr0.0001/ckpts/best_model_epoch_98_rmse_6.0010.pth')
    ap.add_argument("--batch_size", type=int, default=1)  
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--clamp", action="store_true", default=False)

    # transform
    ap.add_argument("--transform_name", choices=["vit","cnn","albumentations","none", 'imagenet'], default="imagenet")
    ap.add_argument("--img_size", type=int, default=224)

    # 输出
    ap.add_argument("--out_dir", default='./Results_infer_HGF', help="输出目录")
    ap.add_argument("--pred_filename", default="test_pred.xlsx")
    ap.add_argument("--tar_filename", default="test_tar.xlsx")

    args = ap.parse_args()

    # 解析 ids
    if args.mode == "first_n":
        picked_ids = _pick_ids(args.hgf_dir, "first_n", args.n, None)
    else:
        id_list = []
        if args.ids: id_list += [s.strip() for s in args.ids.split(",") if s.strip()]
        if args.ids_file:
            with open(args.ids_file, "r") as f:
                id_list += [line.strip() for line in f if line.strip()]
        picked_ids = _pick_ids(args.hgf_dir, "ids", 0, id_list)
    print(f"[HGF] picked {len(picked_ids)} samples.")


    # transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if args.transform_name == 'vit':
        transform = get_vit_transform(args.img_size)
    elif args.transform_name == 'cnn':
        transform = get_cnn_transform(args.img_size, mean, std)
    elif args.transform_name == 'albumentations':
        transform = get_albumentations_transform(args.img_size)
    elif args.transform_name == 'none':
        transform = None
    elif args.transform_name == 'imagenet':
        transform = get_imagenet_transform(args.img_size)
    else:
        raise ValueError(f"Unknown transform type: {args.transform}")

    # DataLoader：只用 HarvardGFDataset + 顺序 sampler
    test_loader = _get_hgf_loader_subset(
        hgf_dir=args.hgf_dir,
        modality_type=args.modality_type,
        task=args.task,
        resolution=args.resolution,
        depth=args.depth,
        transform=transform,
        picked_ids=picked_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 评估所需列
    all_test_id  = picked_ids
    all_test_lat = ["Unknown"] * len(picked_ids)  # 若能解析左右眼，这里替换

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # === 推理（沿用 evaluate_and_generate_predictions）===
    # model = timm.create_model('vit_base_patch8_224.augreg2_in21k_ft_in1k', pretrained=False)
    # num_classes = 52
    # dim = model.head.in_features
    # model.head = nn.Sequential(nn.ReLU(), nn.Linear(dim, num_classes))

    num_classes = 52  

    if args.backbone == 'vit':
        model = timm.create_model(
            'vit_base_patch8_224.augreg2_in21k_ft_in1k',
            pretrained=False,
            pretrained_cfg_overlay=dict(file='/mnt/sda/sijiali/GlaucomaCode/vit_base_patch8_224.augreg2_in1k_ft_in1k/pytorch_model.bin')
        )
        dim = model.head.in_features
        model.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )
    
    elif args.backbone == 'dinov3':
        model = DinoV3Backbone52(
            hf_model_name=args.dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool="cls"             
        )

    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    pred_df = evaluate_and_generate_predictions(
        model=model,
        weight_path=args.weight_path,
        test_loader=test_loader,
        device=device,
        all_test_id=all_test_id,
        all_test_lat=all_test_lat,
        clamp=args.clamp,
    )

    
    os.makedirs(args.out_dir, exist_ok=True)
    pred_xlsx = os.path.join(args.out_dir, args.pred_filename)
    pred_df.to_excel(pred_xlsx, index=False)
    print(f"[SAVE] predictions -> {pred_xlsx}")

    tar_df = _build_tar_df_from_npz(args.hgf_dir, picked_ids, lat_fill=all_test_lat[0])
    tar_xlsx = os.path.join(args.out_dir, args.tar_filename)
    tar_df.to_excel(tar_xlsx, index=False)
    print(f"[SAVE] targets -> {tar_xlsx}")

if __name__ == "__main__":
    main()
