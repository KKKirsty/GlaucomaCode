import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from monai.transforms import MapTransform
from monai.data import MetaTensor
import albumentations
from skimage import io
import numpy as np
import torch
import os
from monai.metrics import RMSEMetric, MAEMetric, SSIMMetric
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRangeD, ResizeD, NormalizeIntensityd, ToTensord
)
import glob
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader
from dataset import HarvardGFDataset, SalsaHGFAlignedDataset
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import math






def set_train_scope(model, scope: str):
    if scope == "head":
        # 先全冻
        for p in model.parameters():
            p.requires_grad = False
        # 只开头部
        for p in model.head.parameters():
            p.requires_grad = True
    elif scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("scope 只能是 'head' 或 'all'")


def save_model(model, save_dir, epoch, mean_rmse, max_ckpts=3):
    filename = f"best_model_epoch_{epoch}_rmse_{mean_rmse:.4f}.pth"
    save_path = os.path.join(save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"New best model saved: {save_path}")

    # 删除多余的 ckpt，只保留最近 max_ckpts 个
    ckpts = sorted(glob.glob(os.path.join(save_dir, "best_model_epoch_*.pth")), key=os.path.getmtime)
    if len(ckpts) > max_ckpts:
        for ckpt in ckpts[:-max_ckpts]:
            os.remove(ckpt)
    return save_path


# def visualize_training_data(loader, num_images=9, save_path="samples.png"):
#     fig, axes = plt.subplots(3, 3, figsize=(5,5))
#     axes = axes.ravel()
#     for i, batch in enumerate(loader):
#         images = batch["image"]
#         for j in range(len(images)):
#             if i * len(images) + j >= num_images:
#                 break
#             img = images[j].numpy().transpose(1, 2, 0)
#             # img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
#             axes[i * len(images) + j].imshow(img)
#             axes[i * len(images) + j].axis('off')
#         print('width and height are: ', img.shape[1], img.shape[0])
#         if i * len(images) >= num_images - 1:
#             break

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved {num_images} sample images to {save_path}")

# === 反标准化（ImageNet） ===
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def denorm_img(t: torch.Tensor) -> torch.Tensor:
    """
    t: (N,3,H,W) or (3,H,W), 已按 ImageNet 做过 Normalize 的 tensor
    return: 同 shape，值域约回到 [0,1]
    """
    single = (t.dim() == 3)
    if single:
        t = t.unsqueeze(0)
    t = t * IMAGENET_STD.to(t.device) + IMAGENET_MEAN.to(t.device)
    t = t.clamp(0, 1)
    return t[0] if single else t

def chw_to_hwc(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) torch.float in [0,1]
    return: (H,W,3) numpy.float in [0,1]
    """
    return x.permute(1,2,0).detach().cpu().numpy()

def make_overlay(base_rgb: torch.Tensor, slab_rgb: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    """
    base_rgb: (3,H,W) 反标准化后的 RNFLT
    slab_rgb: (3,H,W) 反标准化后的 SLAB（其实 3 通道相同）
    用 slab 的灰度作为热度，叠加到 base 上（红色通道增强）。
    返回 (3,H,W)，范围 [0,1]
    """
    # 取 slab 的单通道强度（0~1）
    slab_gray = slab_rgb.mean(dim=0, keepdim=True)  # (1,H,W)
    # 构造红色热度层：R=slab_gray, G=B=0
    red_layer = torch.zeros_like(base_rgb)
    red_layer[0] = slab_gray[0]  # R 通道
    # alpha 混合
    out = (1 - alpha) * base_rgb + alpha * red_layer
    # 让背景仍能看到原图：再与 base 做个最大值混合，或简单 clamp
    out = torch.max(out, base_rgb*0.7).clamp(0,1)
    return out

def visualize_training_data(
    loader,
    num_images: int = 9,
    mode: str = "rnflt",   # "rnflt" | "slab" | "overlay" | "triplet"
    save_path: str = "samples.png"
):
    """
    mode:
      - "rnflt": 只画 RNFLT
      - "slab":  只画 SLAB（三通道）
      - "overlay": RNFLT 底图 + SLAB 热度叠加
      - "triplet": 每个样本画三幅（RNFLT / SLAB / OVERLAY），注意 num_images 会被取到能整除3的数
    """
    images_shown = 0
    panels_per_sample = 3 if mode == "triplet" else 1
    num_panels = (num_images // panels_per_sample) * panels_per_sample
    if num_panels == 0:
        print("num_images 太小啦，至少 >= 1"); return

    # 自动布局
    cols = 3
    rows = math.ceil(num_panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(-1)

    ax_idx = 0
    for batch in loader:
        # batch["image"]: (B,3,H,W)，已标准化
        img = batch["image"]
        has_slab = ("slab" in batch)
        slab = batch["slab"] if has_slab else None

        B = img.shape[0]
        for b in range(B):
            if images_shown >= num_panels:
                break

            # 取单个样本 & 反标准化
            img_b = denorm_img(img[b])  # (3,H,W), 0~1
            if has_slab:
                slab_b = denorm_img(slab[b])  # (3,H,W), 0~1
            else:
                slab_b = None

            if mode == "rnflt":
                axes[ax_idx].imshow(chw_to_hwc(img_b))
                axes[ax_idx].set_title("RNFLT")
                axes[ax_idx].axis('off'); ax_idx += 1; images_shown += 1

            elif mode == "slab":
                if slab_b is None:
                    axes[ax_idx].text(0.5, 0.5, "no slab", ha='center', va='center')
                else:
                    axes[ax_idx].imshow(chw_to_hwc(slab_b))
                axes[ax_idx].set_title("SLAB")
                axes[ax_idx].axis('off'); ax_idx += 1; images_shown += 1

            elif mode == "overlay":
                if slab_b is None:
                    axes[ax_idx].imshow(chw_to_hwc(img_b))
                    axes[ax_idx].set_title("RNFLT (no slab)")
                else:
                    overlay = make_overlay(img_b, slab_b, alpha=0.45)
                    axes[ax_idx].imshow(chw_to_hwc(overlay))
                    axes[ax_idx].set_title("RNFLT + SLAB overlay")
                axes[ax_idx].axis('off'); ax_idx += 1; images_shown += 1

            elif mode == "triplet":
                # RNFLT
                axes[ax_idx].imshow(chw_to_hwc(img_b))
                axes[ax_idx].set_title("RNFLT"); axes[ax_idx].axis('off'); ax_idx += 1
                # SLAB
                if slab_b is None:
                    axes[ax_idx].text(0.5, 0.5, "no slab", ha='center', va='center')
                else:
                    axes[ax_idx].imshow(chw_to_hwc(slab_b))
                axes[ax_idx].set_title("SLAB"); axes[ax_idx].axis('off'); ax_idx += 1
                # OVERLAY
                if slab_b is None:
                    axes[ax_idx].imshow(chw_to_hwc(img_b))
                    axes[ax_idx].set_title("RNFLT (no slab)")
                else:
                    overlay = make_overlay(img_b, slab_b, alpha=0.45)
                    axes[ax_idx].imshow(chw_to_hwc(overlay))
                    axes[ax_idx].set_title("Overlay")
                axes[ax_idx].axis('off'); ax_idx += 1; images_shown += 3

        if images_shown >= num_panels:
            break

    # 清理多余子图
    for k in range(ax_idx, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {images_shown} panel(s) to {save_path}")


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y) + 1e-8)  # 避免nan



# ======================== Transform ======================== #
# def get_imagenet_transform(img_size):
#     return Compose([
#         ResizeD(keys=['image'], spatial_size=(img_size, img_size)),
#         # 如果图像确实是 0~255 亮度，先缩放到 [0,1]
#         ScaleIntensityRangeD(keys=['image'],
#                              a_min=0.0, a_max=255.0,
#                              b_min=0.0, b_max=1.0, clip=True),

#         # 固定通道均值/方差（ImageNet）
#         NormalizeIntensityd(
#             keys=['image'],
#             subtrahend=[0.485, 0.456, 0.406],
#             divisor=[0.229, 0.224, 0.225],
#             channel_wise=True
#         ),

#         ToTensord(keys=['image', 'label']),
#     ])

def get_imagenet_transform(img_size, with_slab=False):
    keys = ['image'] + (['slab'] if with_slab else [])
    return Compose([
        # 0–255 -> [0,1]
        ScaleIntensityRangeD(
            keys=keys,
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0, clip=True
        ),
        # Resize 到 DINOv3 输入（例如 224、336）
        # ResizeD 的 mode 可以给单个字符串或与 keys 对应的列表
        ResizeD(
            keys=keys,
            spatial_size=(img_size, img_size),
            mode='bilinear'  # 两个 key 都是图像，双线性即可
        ),
        # 按 ImageNet 规范做标准化
        NormalizeIntensityd(
            keys=keys,
            subtrahend=[0.485, 0.456, 0.406],
            divisor=[0.229, 0.224, 0.225],
            channel_wise=True
        ),
        ToTensord(keys=keys + ['label']),
    ])


def get_vit_transform(img_size):
    return Compose([
        ResizeD(keys=['image'], spatial_size=(img_size, img_size)),
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=['image', 'label'])
    ])

def get_cnn_transform(img_size, mean, std):
    return Compose([
        ResizeD(keys=['image'], spatial_size=(img_size, img_size)),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=mean,
            divisor=std,
            channel_wise=True
        ),
        ToTensord(keys=['image', 'label'])
    ])


class AlbumentationsTransform(MapTransform):
    def __init__(self, keys, albumentations_aug):
        super().__init__(keys)
        self.albumentations_aug = albumentations_aug

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            # 如果是 torch.Tensor，先转成 numpy
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            # [3, H, W] 转 [H, W, 3]；[1, H, W] 转 [H, W, 1]；[H, W] 转 [H, W, 1]
            if image.ndim == 3:
                if image.shape[0] == 1 or image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
            elif image.ndim == 2:
                image = np.expand_dims(image, -1)
            # 进入 albumentations (输入 [H, W, C])
            augmented = self.albumentations_aug(image=image)
            image_aug = augmented['image']
            # 输出 [C, H, W]
            if image_aug.ndim == 3:
                image_aug = np.transpose(image_aug, (2, 0, 1)).astype(np.float32)
            elif image_aug.ndim == 2:
                image_aug = np.expand_dims(image_aug, 0).astype(np.float32)
            image_aug = torch.tensor(image_aug, dtype=torch.float)
            data[key] = MetaTensor(image_aug, affine=None)
        return data

def get_albumentations_transform(img_size):
    albumentations_transform = AlbumentationsTransform(
        keys=["image"],
        albumentations_aug=albumentations.Compose([
            albumentations.Resize(img_size, img_size, p=1),
            albumentations.UnsharpMask(p=1),
            # albumentations.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0, p=1.0),
        ])
    )
    return Compose([
        albumentations_transform,
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=['label']),
    ])






# ======================== Infer ======================== #
def evaluate_and_generate_predictions(model, weight_path, test_loader, device, all_test_id, all_test_lat, clamp = False):

    rmse_metric = RMSEMetric()
    mae_metric = MAEMetric()
    rmse_metric.reset()
    mae_metric.reset()

    all_data = []  # List to store predictions along with RMSE and MAE
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
            pred = model(test_images)
            pred = pred * 10.0
            if clamp:
                pred = torch.clamp(pred, min=0)

            rmse_metric(pred, test_labels)
            mae_metric(pred, test_labels)

            image_rmse = rmse_metric(pred, test_labels).item()
            image_mae = mae_metric(pred, test_labels).item()
            mean_rmse = rmse_metric.aggregate().item()
            mean_mae = mae_metric.aggregate().item()

            print(f'Test Mean RMSE: {mean_rmse:.4f}, Test MAE: {mean_mae:.4f}')
            all_data.append(list(pred.cpu().numpy()[0]) + [image_rmse, image_mae])

    columns = [f'pred_{i}' for i in range(1, 53)] + ['image_rmse', 'image_mae']
    pred_df = pd.DataFrame(all_data, columns=columns)
    pred_df.insert(0, 'test_id', all_test_id)
    pred_df.insert(1, 'test_lat', all_test_lat)

    return pred_df



# ======================== Visualization for SALSA ======================== #
# ==== Helpers: list/pick ids ====
def list_all_ids(hgf_dir: str) -> List[str]:
    p = Path(hgf_dir)
    ids = sorted([f.name for f in p.iterdir() if f.is_dir()])
    if not ids:
        raise RuntimeError(f'No subdirectories found in {hgf_dir}')
    return ids

def pick_ids(hgf_dir: str, mode: str, n: int = 0, ids: Optional[List[str]] = None) -> List[str]:
    all_ids = list_all_ids(hgf_dir)
    if mode == 'first_n':
        assert n > 0, 'n must be > 0 for mode=first_n'
        return all_ids[:n]
    elif mode == 'ids':
        assert ids and len(ids) > 0, 'mode=ids requires ids'
        aset = set(all_ids)
        return [i for i in ids if i in aset]
    else:
        raise ValueError(f'Unknown mode: {mode}')

# ==== Build target DataFrame from .npz ====
def build_tar_df_from_npz(hgf_dir: str, picked_ids: List[str], lat_fill: str = 'Unknown') -> pd.DataFrame:
    vecs = []
    for sid in picked_ids:
        npz_path = os.path.join(hgf_dir, f'{sid}.npz')
        raw = np.load(npz_path, allow_pickle=True)
        if 'tds' not in raw.files:
            raise KeyError(f"'tds' not in {npz_path}, keys={raw.files}")
        vecs.append(raw['tds'].astype(np.float32))
    mat = np.stack(vecs, axis=0)  # (N,K)
    cols = [f'tar_{i}' for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, 'test_id', picked_ids)
    df.insert(1, 'test_lat', [lat_fill] * len(picked_ids))
    return df


# ==== Sequential sampler on specific indices ====
class SeqOnIndices(torch.utils.data.Sampler):
    def __init__(self, indices: List[int]):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

# ==== DataLoader subset ====
# def get_hgf_loader_subset(
#     hgf_dir: str,
#     modality_type: str,
#     task: str,
#     resolution: int,
#     depth: int,
#     transform,
#     picked_ids: List[str],
#     batch_size: int = 1,
#     num_workers: int = 4,
# ) -> DataLoader:
#     ds = HarvardGFDataset(
#         data_path=hgf_dir,
#         modality_type=modality_type,
#         task=task,
#         resolution=resolution,
#         depth=depth,
#         transform=transform,
#     )
#     id2idx = {Path(f).stem: i for i, f in enumerate(ds.files)}
#     indices = [id2idx[s] for s in picked_ids if s in id2idx]
#     sampler = SeqOnIndices(indices)
#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         sampler=sampler,
#         pin_memory=True,
#     )

def get_salsa_loader_subset(
    image_root: str,
    hgf_test_root: str,
    modality_type: str,
    transform,
    picked_ids: List[str],
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    ds = SalsaHGFAlignedDataset(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        transform=transform
    )

    id2idx = {sid: i for i, sid in enumerate(ds.sample_ids)}
    indices = [id2idx[sid] for sid in picked_ids if sid in id2idx]

    sampler = SeqOnIndices(indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )


# ==== 52-point layout & visualization ====
HARVARD_GF_52_LAYOUT = [
    [None,  0,  1,  2,  3,  4,  5, None],
    [  6,   7,  8,  9, 10, 11, 12, 13],
    [ 14,  15, 16, 17, 18, 19, 20, 21],
    [ 22,  23, 24, 25, 26, 27, 28, 29],
    [ 30,  31, 32, 33, 34, 35, 36, 37],
    [ 38,  39, 40, 41, 42, 43, 44, 45],
    [None, 46, 47, 48, 49, 50, 51, None],
]

def _draw_vf(ax, values, layout=HARVARD_GF_52_LAYOUT, vmin=None, vmax=None,
             cmap='gray', fontsize=8, fmt='{:.1f}'):
    values = np.asarray(values).reshape(-1)
    assert values.size == 52, f'Expect 52 values, got {values.size}'
    if vmin is None: vmin = float(np.nanmin(values))
    if vmax is None: vmax = float(np.nanmax(values))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.set_aspect('equal'); ax.axis('off')
    n_rows, n_cols = len(layout), len(layout[0])
    for r in range(n_rows):
        for c in range(n_cols):
            idx = layout[r][c]
            if idx is None: continue
            v = float(values[idx])
            rect = Rectangle((c, r), 1, 1, facecolor=plt.get_cmap(cmap)(norm(v)),
                             edgecolor='k', linewidth=0.5)
            ax.add_patch(rect)
            lum = norm(v)
            txt_color = 'white' if lum < 0.4 else 'black'
            ax.text(c+0.5, r+0.5, fmt.format(v), ha='center', va='center',
                    fontsize=fontsize, color=txt_color)
    ax.set_xlim(0, n_cols); ax.set_ylim(n_rows, 0)

def plot_vf_triptych(input_img, gt_52, pred_52, layout=HARVARD_GF_52_LAYOUT,
                     vmin=None, vmax=None, save_path=None, titles=('Input','Ground truth','Prediction')):
    gt_52  = np.asarray(gt_52).reshape(-1)
    pred_52 = np.asarray(pred_52).reshape(-1)
    if vmin is None: vmin = float(np.nanmin([gt_52.min(), pred_52.min()]))
    if vmax is None: vmax = float(np.nanmax([gt_52.max(), pred_52.max()]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios':[1.2,1,1]})
    # Input
    ax0 = axes[0]; ax0.axis('off'); ax0.set_title(titles[0])
    img = input_img
    if torch.is_tensor(img): img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1,3): img = np.transpose(img, (1,2,0))
    if img.ndim == 2: ax0.imshow(img, cmap='gray')
    else: ax0.imshow(np.clip(img, 0, 1))
    # GT & Pred
    axes[1].set_title(titles[1]); _draw_vf(axes[1], gt_52, layout, vmin, vmax)
    axes[2].set_title(titles[2]); _draw_vf(axes[2], pred_52, layout, vmin, vmax)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200); plt.close(fig)
    else: plt.show()

def preview_layout(layout=HARVARD_GF_52_LAYOUT):
    dummy = np.arange(52)
    fig, ax = plt.subplots(figsize=(4,4))
    _draw_vf(ax, dummy, layout, vmin=0, vmax=51, fmt='{:.0f}')
    ax.set_title('Index layout (0..51)'); plt.tight_layout(); plt.show()



def _strip_module_prefix(state_dict):
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def visualize_after_training(
    val_loader,
    model,
    device,
    out_dir,
    weight_path: Optional[str] = None,    
    pred_filename: str = "pred.xlsx",
    tar_filename: str = "targets.xlsx",
    num_classes: int = 52,
    scale: float = 10.0,
    visualize_n: int = 12,             
    export_targets: bool = False,
    hgf_test_root: Optional[str] = None,
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 加载最佳权重
    if weight_path:
        ckpt = torch.load(weight_path, map_location='cpu')
        state = ckpt.get('model_state', ckpt)
        state = _strip_module_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print('[load] missing:', missing, '| unexpected:', unexpected)

    model.eval(); model.to(device)
    pred_rows, imgs_cache = [], []
    ids_used = []  # 仅记录前 N 个样本的 id

    # 准备 ID/lat
    all_ids = getattr(val_loader.dataset, 'ids', None)
    if all_ids is None:
        all_ids = [f"sample_{i}" for i in range(len(val_loader.dataset))]
    all_lat = ['Unknown'] * len(all_ids)

    # 整体聚合，只统计前 N 个
    try:
        from monai.metrics import RMSEMetric, MAEMetric
        rmse_metric = RMSEMetric(); mae_metric = MAEMetric()
        use_monai = True
    except Exception:
        use_monai = False

    processed = 0
    idx = 0
    with torch.no_grad():
        for batch in val_loader:
            if processed >= visualize_n:
                break  # 外层提前结束

            x = batch['image'].to(device)
            y = batch.get('label', None)

            out_scaled = model(x)          # [B, 52]
            out = out_scaled * scale

            out_np = out.detach().cpu().numpy()
            B = out_np.shape[0]

            for b in range(B):
                if processed >= visualize_n:
                    break  # 内层提前结束

                sid = all_ids[idx] if idx < len(all_ids) else f"sample_{idx}"
                row = {'test_id': sid, 'test_lat': all_lat[idx] if idx < len(all_lat) else 'Unknown'}

                # 写入 52 维预测
                for k in range(num_classes):
                    row[f'pred_{k}'] = float(out_np[b, k])

                # 逐样本 MAE/RMSE
                if y is not None:
                    gt_b = y[b].cpu().numpy()
                    diff = out_np[b] - gt_b
                    mae  = float(np.mean(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(diff ** 2)))

                    if use_monai:
                        rmse_metric(out[b:b+1], y[b:b+1].to(device))  # 只累加这一张
                        mae_metric(out[b:b+1],  y[b:b+1].to(device))
                else:
                    mae = rmse = float('nan')

                row['mae'] = mae
                row['rmse'] = rmse
                pred_rows.append(row)
                ids_used.append(sid)

                # 打印 + 缓存（这时“可视化数量”与“验证数量”一致）
                print(f"[VAL] id={sid}  MAE={mae:.4f}  RMSE={rmse:.4f}")
                img = batch['image'][b]  # (C,H,W)
                gt_arr = (y[b].cpu().numpy() if y is not None else np.full(num_classes, np.nan))
                imgs_cache.append((sid, img, gt_arr, out_np[b]))

                processed += 1
                idx += 1

            # 如果这一批没用完的样本也要推进 idx：
            while idx < len(all_ids) and idx < processed:
                idx += 1

    # 导出（只包含前 N 个样本）
    pred_df = pd.DataFrame(pred_rows)
    pred_xlsx = os.path.join(out_dir, pred_filename)
    pred_df.to_excel(pred_xlsx, index=False)
    print(f'[SAVE] predictions -> {pred_xlsx}')

    # 整体平均（仅前 N 个）
    if use_monai and processed > 0:
        try:
            overall_rmse = rmse_metric.aggregate().item()
            overall_mae  = mae_metric.aggregate().item()
            print(f"[VAL] (first {processed}) Overall RMSE: {overall_rmse:.4f}, Overall MAE: {overall_mae:.4f}")
        except Exception:
            pass

    # # 导出 targets（仅前 N 个）
    # if export_targets and hgf_test_root is not None and ids_used:
    #     try:
    #         tar_df = build_tar_df_from_npz(hgf_test_root, ids_used, lat_fill='Unknown')
    #         tar_xlsx = os.path.join(out_dir, tar_filename)
    #         tar_df.to_excel(tar_xlsx, index=False)
    #         print(f'[SAVE] targets -> {tar_xlsx}')
    #     except Exception as e:
    #         print(f"[WARN] export targets failed: {e}")

    # Visualization:
    vmin, vmax = None, None 

    save_pngs = True
    save_dir = os.path.join(out_dir, 'figs')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for sid, img, gt, pred_scaled in imgs_cache:
        pred = pred_scaled * 1.0   # already scaled back earlier (out = out_scaled*scale)
        save_path = os.path.join(save_dir, f'{sid}.png') if save_pngs else None
        plot_vf_triptych(
            input_img=img, 
            gt_52=gt, 
            pred_52=pred, 
            vmin=vmin, vmax=vmax, 
            save_path=save_path
        )
        if not save_pngs:
            break  # if not saving, display only the first one
    print(f'[SAVE] figures -> {save_dir}')



