import argparse
import os
import numpy as np
import json
from utils import get_vit_transform, get_cnn_transform, get_albumentations_transform, get_imagenet_transform
from train import worker_salsa_multi

def main():
    parser = argparse.ArgumentParser(description='SALSA-derived RNFL+slab training with ViT')
    # model
    parser.add_argument('--backbone', type=str, default='dinov3', choices=['dinov3'], help='Backbone: dinov3')
    parser.add_argument('--dinov3_model', type=str,
                    default='/mnt/sda/sijiali/GlaucomaCode/pretrained_weight/dinov3-vitb16-pretrain-lvd1689m',
                    help='HuggingFace DINOv3 model name or path')
    parser.add_argument('--vit_pool', type=str, default='cls', choices=['cls', 'mean_patch'], help='ViT feature pooling method: cls or mean_patch')
    parser.add_argument('--train_scope', type=str, default='all', choices=['head', 'all'], help='Training scope: head or all')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'gated-sum', 'sum', 'attn'], help='Feature fusion method for dual-branch model')
    # dataset
    parser.add_argument('--data_root', type=str, default='/mnt/sda/sijiali/DataSet/harvardGF_unpacked',
                        help='SALSA images root. Subfolders like data_xxxx contain rnfl_thickness_map.jpg and optional slab_image.png')
    parser.add_argument('--hgf_test_root', type=str, default='/mnt/sda/sijiali/DataSet/Harvard-GF/Dataset/Test',
                        help='Harvard-GF Test root containing data_xxxx.npz with key "tds"')
    parser.add_argument('--modality_type', type=str, default='rnflt+slab', choices=['rnflt', 'rnflt+slab'],
                        help='Use RNFL only or RNFL+SLAB')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Split ratio for validation if no split file')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--task', type=str, default='tds', choices=['tds'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_txt', type=str, default=None)
    # training
    parser.add_argument('--train_batch', type=int, default=32)
    parser.add_argument('--valid_batch', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--lr', type=str, default='1e-4')
    parser.add_argument('--transform', type=str, choices=['vit', 'cnn', 'albumentations', 'none', 'imagenet'], default='imagenet')
    # device
    parser.add_argument('--gpus', type=str, default='0', help='Comma separated GPU ids, e.g., "0,2,3"')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')

    args = parser.parse_args()

    # 设定可见 GPU
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print('Using GPU(s):', args.gpus)

    # 创建目录
    base_dir = f"./Results_SALSA_multi/{args.backbone}"
    run_tag  = f"{args.backbone}_{args.transform}_{args.train_scope}_{args.vit_pool}_{args.fusion}_lr{args.lr}"
    run_root = os.path.join(base_dir, run_tag)

    args.save_dir = os.path.join(run_root, "ckpts")
    args.log_dir  = os.path.join(run_root, "logs")
    visual_dir = os.path.join(run_root, "visualization")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir, exist_ok=True)

    with open(os.path.join(run_root, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # transform
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    if args.transform == 'vit':
        base_transform = get_vit_transform(args.img_size)
    elif args.transform == 'cnn':
        base_transform = get_cnn_transform(args.img_size, mean, std)
    elif args.transform == 'albumentations':
        base_transform = get_albumentations_transform(args.img_size)
    elif args.transform == 'none':
        base_transform = None
    elif args.transform == 'imagenet':
        base_transform = get_imagenet_transform(args.img_size, with_slab=(args.modality_type=='rnflt+slab'))
    else:
        raise ValueError(f'Unknown transform type: {args.transform}')

    # 训练
    worker_salsa_multi(
        image_root=args.data_root,
        hgf_test_root=args.hgf_test_root,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        visual_dir=visual_dir,
        transform=base_transform,   
        lr=float(args.lr), device=args.device, max_epochs=args.epochs, val_interval=args.val_interval,
        train_batch=args.train_batch, valid_batch=args.valid_batch, num_workers=args.num_workers, patience=args.patience,
        modality_type=args.modality_type, 
        val_ratio=args.val_ratio, split_txt=args.split_txt, seed=args.seed,
        backbone=args.backbone, dinov3_model=args.dinov3_model, vit_pool=args.vit_pool, train_scope=args.train_scope, fusion=args.fusion,
    )

if __name__ == '__main__':
    main()
