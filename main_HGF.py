import argparse
import os
import numpy as np
from utils import get_vit_transform, get_cnn_transform, get_albumentations_transform, get_imagenet_transform
import json



def main():

    parser = argparse.ArgumentParser(description='Harvard-GF VF prediction with ViT')
    # model
    parser.add_argument('--backbone', type=str, default='dinov3',
                    choices=['vit', 'dinov3'], help='Backbone: vit or dinov3')
    parser.add_argument('--dinov3_model', type=str,
                    default='/mnt/sda/sijiali/GlaucomaCode/pretrained_weight/dinov3-vitb16-pretrain-lvd1689m',
                    help='HuggingFace DINOv3 model name or path')
    parser.add_argument('--vit_pool', type=str, default='cls', choices=['cls', 'mean_patch'], help='ViT feature pooling method: cls or mean_patch')
    parser.add_argument('--train_scope', type=str, default='head', choices=['head', 'all'], help='Training scope: head or all')
    
    # dataset
    parser.add_argument('--train_dir', default='/mnt/sda/sijiali/DataSet/Harvard-GF/Dataset/Training', type=str)
    parser.add_argument('--val_dir',   default='/mnt/sda/sijiali/DataSet/Harvard-GF/Dataset/Validation', type=str)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--modality_type', type=str, default='rnflt', choices=['rnflt', 'bscan'], help='Modality type: rnflt or bscan')
    parser.add_argument('--task', type=str, default='tds', choices=['tds'], help='Task type: tds')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the RNFLT image stack, 1 for single image, >1 for multiple images')
    # training
    parser.add_argument('--train_batch', type=int, default=32)
    parser.add_argument('--valid_batch', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--lr', type=str, default='5e-3')
    parser.add_argument('--transform', type=str, choices=['vit', 'cnn', 'albumentations', 'none', 'imagenet'], default='imagenet')
    parser.add_argument('--save_dir', default='./Results/vit_transform_lr1e-3/ckpts', type=str, help="Directory to save model checkpoints")
    parser.add_argument('--log_dir', default='./Results/vit_transform_lr1e-3/logs', type=str, help="Directory to save training logs")
    # device
    parser.add_argument('--gpus', type=str, default='0', help='Comma separated GPU ids, e.g., "0,2,3"')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    args = parser.parse_args()

 
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print("Using GPU(s):", args.gpus)
    from train import worker_HGF


    # 创建目录
    base_dir = f"./Results_HGF/{args.backbone}"
    run_tag  = f"{args.backbone}_{args.transform}_{args.vit_pool}_{args.train_scope}_lr{args.lr}"
    run_root = os.path.join(base_dir, run_tag)

    args.save_dir = os.path.join(run_root, "ckpts")
    args.log_dir  = os.path.join(run_root, "logs")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(run_root, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


    # transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if args.transform == 'vit':
        transform = get_vit_transform(args.img_size)
    elif args.transform == 'cnn':
        transform = get_cnn_transform(args.img_size, mean, std)
    elif args.transform == 'albumentations':
        transform = get_albumentations_transform(args.img_size)
    elif args.transform == 'none':
        transform = None
    elif args.transform == 'imagenet':
        transform = get_imagenet_transform(args.img_size)
    else:
        raise ValueError(f"Unknown transform type: {args.transform}")

    # training
    worker_HGF(
        args.train_dir, args.val_dir, args.save_dir, args.log_dir,
        transform, float(args.lr), args.device, args.epochs, args.val_interval,
        args.train_batch, args.valid_batch, args.num_workers, args.patience,
        modality_type=args.modality_type, task=args.task, resolution=args.img_size, depth= args.depth, 
        backbone=args.backbone, dinov3_model=args.dinov3_model, vit_pool=args.vit_pool, train_scope=args.train_scope
    )



if __name__ == '__main__':
    main()
