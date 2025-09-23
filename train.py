from torch.utils.tensorboard import SummaryWriter
from utils import RMSELoss,set_train_scope, visualize_after_training
import torch
from monai.metrics import RMSEMetric, MAEMetric, SSIMMetric
from torch.optim import lr_scheduler
import timm
from torch.optim import lr_scheduler, Adam
from dataset import get_dataloader_HGF, get_dataloader_Salsa, SalsaHGFAlignedDataset
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import visualize_training_data, save_model
import os
import random
from dino_model import DinoV3Backbone52





def train_model(train_loader, valid_loader, model, optimizer, max_epochs, scheduler, val_interval, device, save_dir, log_dir, patience=50):

    writer = SummaryWriter(log_dir)
    loss_function = RMSELoss()
    rmse_metric = RMSEMetric()
    mae_metric = MAEMetric()
    best_metric = float('inf')
    best_metric_epoch = -1
    best_metric_mae = None
    best_ckpt_path = None
    epoch_loss_values = []
    counter = 0

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            labels = labels / 10.0  # Normalize the targets
            optimizer.zero_grad()
            outputs_scaled = model(inputs)
            loss = loss_function(outputs_scaled, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # PyTorch built-in scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                # Custom CosineScheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)
                current_lr = scheduler.get_last_lr()[0]

        # Logging
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        print(f"epoch {epoch + 1} Epoch average loss: {epoch_loss:.4f}, Current LR: {current_lr:.8f}")

        # Validation phase
        if (epoch + 1) % val_interval == 0:
            model.eval()
            rmse_metric.reset()
            mae_metric.reset()
            with torch.no_grad():
                for val_data in valid_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_images)
                    # Denormalize the outputs
                    val_outputs = val_outputs * 10.0
                    # Update the metrics with the current batch
                    rmse_metric(val_outputs, val_labels)
                    mae_metric(val_outputs, val_labels)

                # Compute the mean RMSE and MAE over the validation set
                mean_rmse = rmse_metric.aggregate().item()
                mean_mae = mae_metric.aggregate().item()

                # Logging and model saving logic
                writer.add_scalar('Validation RMSE', mean_rmse, epoch)
                writer.add_scalar('Validation MAE', mean_mae, epoch)
                print(f'Epoch {epoch + 1}, Validation Mean RMSE: {mean_rmse:.4f}, Validation MAE: {mean_mae:.4f}')

                if mean_rmse < best_metric:
                    best_metric = mean_rmse
                    best_metric_epoch = epoch + 1
                    best_metric_mae = mean_mae
                    best_ckpt_path = save_model(model, save_dir, epoch + 1, mean_rmse)
                    #print(f"Saved new best model weights to {weight_file_path}")
                    print(f"New best model saved at epoch {epoch + 1} with Mean RMSE: {mean_rmse:.4f}")
                    # Reset the metrics for the next epoch
                    rmse_metric.reset()
                    mae_metric.reset()
                    counter = 0
                else:
                    counter += 1
                    rmse_metric.reset()
                    mae_metric.reset()
                    if counter >= patience:
                        print(f"Validation MAE did not improve for {patience} consecutive epochs. Early stopping.")
                        break  # Stop training early
    writer.close()
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, mae: {best_metric_mae:.4f}")

    return best_ckpt_path




# ======================== Harvard-GF ======================== #
def worker_HGF(
    train_dir, val_dir, save_dir, log_dir,
    transform, lr, device, max_epochs, val_interval,
    train_batch, valid_batch, num_workers, patience,
    modality_type='rnflt', task='tds', resolution=224, depth=3,
    backbone='dinov3', dinov3_model='facebook/dinov3-convnext-tiny-pretrain-lvd1689m', 
    vit_pool='cls', train_scope='head'
):
    # Loader
    train_loader = get_dataloader_HGF(train_dir, modality_type, task, resolution, depth, transform=transform, batch_size=train_batch, shuffle=True, num_workers=num_workers)
    val_loader   = get_dataloader_HGF(val_dir, modality_type, task, resolution, depth, transform=transform, batch_size=valid_batch, shuffle=False, num_workers=num_workers)
    visualize_training_data(train_loader)
    print('set dataloader...')

    # Model
    num_classes = 52  

    if backbone == 'vit':
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
        
        set_train_scope(model, train_scope)
    
    elif backbone == 'dinov3':
        model = DinoV3Backbone52(
            hf_model_name=dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool=vit_pool         
        )
        
        set_train_scope(model, train_scope)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    
    
    model = model.to(device)
    model = nn.DataParallel(model)

    print('set model...')

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    # Training loop
    train_model(train_loader, val_loader, model, optimizer, max_epochs, scheduler, val_interval, device, save_dir, log_dir, patience)





# ======================== SALSA (rnflt only) ======================== #

def _make_split_ids(root, val_ratio=0.2, split_txt=None, seed=42):
    all_ids = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    assert all_ids, f"No subfolders under {root}"
    if split_txt and os.path.exists(split_txt):
        with open(split_txt, 'r') as f:
            val_ids = [line.strip() for line in f if line.strip()]
        val_set = set(val_ids)
        train_ids = [x for x in all_ids if x not in val_set]
        return sorted(train_ids), sorted(val_ids)
    rng = random.Random(seed)
    ids = all_ids[:]
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids

def worker_salsa(
    image_root, hgf_test_root, save_dir, log_dir, visual_dir,
    transform, lr, device, max_epochs, val_interval,
    train_batch, valid_batch, num_workers, patience,
    modality_type='rnflt', val_ratio=0.2, split_txt=None, seed=42,
    backbone='dinov3', dinov3_model='facebook/dinov3-convnext-tiny-pretrain-lvd1689m', 
    vit_pool='cls', train_scope='head'
):

    # 1) 拆分 ID
    train_ids, val_ids = _make_split_ids(image_root, val_ratio=val_ratio, split_txt=split_txt, seed=seed)
    print(f"[SALSA split] train={len(train_ids)}, val={len(val_ids)}")

    # 2) 用 get_dataloader_Salsa 构建两个 loader（通过 ids 精确选择子集）
    train_loader = get_dataloader_Salsa(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        transform=transform,
        ids=train_ids,            
    )
    val_loader = get_dataloader_Salsa(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        batch_size=valid_batch,
        shuffle=False,            
        num_workers=num_workers,
        transform=transform,
        ids=val_ids,          
    )

    visualize_training_data(loader=train_loader, save_path=os.path.join(visual_dir, 'train_samples.png'))
    print('set dataloader...')

    # 3) 模型
    num_classes = 52  

    if backbone == 'vit':
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
        set_train_scope(model, train_scope)
    
    elif backbone == 'dinov3':
        model = DinoV3Backbone52(
            hf_model_name=dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool=vit_pool         
        )
        set_train_scope(model, train_scope)
        
        vis_model = DinoV3Backbone52(
            hf_model_name=dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool=vit_pool
        )   
        
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    
    model = model.to(device)
    model = nn.DataParallel(model)
    print('set model...')

    # 4) 优化器 & 调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    # 5) 训练
    best_ckpt_path = train_model(train_loader, val_loader, model, optimizer, max_epochs,
                        scheduler, val_interval, device, save_dir, log_dir, patience)
    
    # 6) 可视化
    print("="*30 + " Visualization " + "="*30)
    visualize_after_training(
        val_loader=val_loader,
        model=vis_model,
        device='cpu',
        out_dir=visual_dir,
        weight_path=best_ckpt_path, 
        num_classes=52,
        scale=10.0,
        visualize_n=12,
        export_targets=True,        
        hgf_test_root=hgf_test_root
    )


# ======================== SALSA (rnflt + slab) ======================== #
def _make_split_ids(root, val_ratio=0.2, split_txt=None, seed=42):
    all_ids = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    assert all_ids, f"No subfolders under {root}"
    if split_txt and os.path.exists(split_txt):
        with open(split_txt, 'r') as f:
            val_ids = [line.strip() for line in f if line.strip()]
        val_set = set(val_ids)
        train_ids = [x for x in all_ids if x not in val_set]
        return sorted(train_ids), sorted(val_ids)
    rng = random.Random(seed)
    ids = all_ids[:]
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids

def worker_salsa_multi(
    image_root, hgf_test_root, save_dir, log_dir, visual_dir,
    transform, lr, device, max_epochs, val_interval,
    train_batch, valid_batch, num_workers, patience,
    modality_type='rnflt', val_ratio=0.2, split_txt=None, seed=42,
    backbone='dinov3', dinov3_model='facebook/dinov3-convnext-tiny-pretrain-lvd1689m', 
    vit_pool='cls', train_scope='head'
):

    # 1) 拆分 ID
    train_ids, val_ids = _make_split_ids(image_root, val_ratio=val_ratio, split_txt=split_txt, seed=seed)
    print(f"[multi-SALSA split] train={len(train_ids)}, val={len(val_ids)}")

    # 2) 用 get_dataloader_Salsa 构建两个 loader（通过 ids 精确选择子集）
    train_loader = get_dataloader_Salsa(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        transform=transform,
        ids=train_ids,            
    )
    val_loader = get_dataloader_Salsa(
        image_root=image_root,
        hgf_test_root=hgf_test_root,
        modality_type=modality_type,
        batch_size=valid_batch,
        shuffle=False,            
        num_workers=num_workers,
        transform=transform,
        ids=val_ids,          
    )

    visualize_training_data(loader=train_loader, num_images=9, mode='triplet', save_path=os.path.join(visual_dir, 'train_samples.png'))
    print('set dataloader...')

    # 3) 模型
    num_classes = 52  
    
    if backbone == 'dinov3':
        model = DinoV3Backbone52(
            hf_model_name=dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool=vit_pool         
        )
        set_train_scope(model, train_scope)
        
        vis_model = DinoV3Backbone52(
            hf_model_name=dinov3_model,
            out_dim=num_classes,
            apply_imagenet_norm=False,   
            vit_pool=vit_pool
        )   
        
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    
    model = model.to(device)
    model = nn.DataParallel(model)
    print('set model...')

    # 4) 优化器 & 调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    # 5) 训练
    best_ckpt_path = train_model(train_loader, val_loader, model, optimizer, max_epochs,
                        scheduler, val_interval, device, save_dir, log_dir, patience)
    
    # 6) 可视化
    print("="*30 + " Visualization " + "="*30)
    visualize_after_training(
        val_loader=val_loader,
        model=vis_model,
        device='cpu',
        out_dir=visual_dir,
        weight_path=best_ckpt_path, 
        num_classes=52,
        scale=10.0,
        visualize_n=12,
        export_targets=True,        
        hgf_test_root=hgf_test_root
    )