from torch.utils.tensorboard import SummaryWriter
from utils import RMSELoss,set_train_scope, visualize_after_training, visualize_after_training_dual
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
from dino_model import DinoV3Backbone52, DualDinoV3LateFusion52
from torchinfo import summary as info_summary



def _unwrap_module(m):
    """拿到可能被 DataParallel/DistributedDataParallel 包裹的真实子模块。"""
    return m.module if hasattr(m, "module") else m

def _model_supports_dual_inputs(model) -> bool:
    """
    粗判：模型是否为“双骨干/双输入”类型。
    - 兼容 DataParallel/DDP 包裹
    - 通过是否存在典型属性来判断（如 backbone_s 或 head 支持 late fusion）
    """
    m = _unwrap_module(model)
    has_dual_backbone = hasattr(m, "backbone_s") or hasattr(m, "slab_backbone")  # 你的命名可能不同
    # 也可根据类名做兜底判断（可选）
    class_name = m.__class__.__name__.lower()
    looks_like_dual = any(k in class_name for k in ["dualdino", "latefusion", "twobranch", "multimodal"])
    return bool(has_dual_backbone or looks_like_dual)

def _forward_model(model, batch_dict, device):
    """
    统一的前向封装：
    - 若 batch 里有 'slab' 且模型支持双输入 → 用关键字 rnflt=..., slab=...
    - 否则按单输入调用 model(x)
    说明：
    - 假设 dataloader 返回的 key 为 'image'（rnflt）与可选的 'slab'
    - 若你的 Dual 模型 forward 接口是 model(rnflt=x, slab=y)，这里会自动匹配
    """
    x = batch_dict["image"].to(device)  # rnflt
    use_dual = _model_supports_dual_inputs(model) and ("slab" in batch_dict)

    if use_dual:
        s = batch_dict["slab"].to(device)
        try:
            # 我们推荐的 Dual 接口：显式关键字
            y = model(rnflt=x, slab=s)
        except TypeError:
            # 兜底：有些实现用了位置参数
            y = model(x, s)
        return y, True
    else:
        # 单输入路径（兼容你之前的模型）
        y = model(x)
        return y, False

def train_model_multi(
    train_loader,
    valid_loader,
    model,
    optimizer,
    max_epochs,
    scheduler,
    val_interval,
    device,
    save_dir,
    log_dir,
    patience=50,
    label_scale=10.0,     
    use_amp=False,        # AMP 加速
):
    """
    同时兼容：
      - 只有 rnflt 的单输入模型（老版）
      - rnflt + slab 的双输入模型（方案 D 等）

    关键改动：
      - 使用 _forward_model() 自动决定 model 的调用方式
      - 训练时对 labels 做 /label_scale；验证时对 outputs 再 *label_scale
      - 记录是否使用 slab 参与训练/验证
    """
    writer = SummaryWriter(log_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_function = RMSELoss()
    rmse_metric = RMSEMetric()
    mae_metric = MAEMetric()

    best_metric = float('inf')
    best_metric_epoch = -1
    best_metric_mae = None
    best_ckpt_path = None
    epoch_loss_values = []
    counter = 0

    model = model.to(device)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0.0
        step = 0

        # 统计这轮是否真的用到了 slab
        used_slab_in_epoch = 0
        batch_count_in_epoch = 0

        for batch_data in train_loader:
            step += 1
            batch_count_in_epoch += 1

            # 准备标签：训练按你原逻辑缩放到更小尺度
            labels = batch_data["label"].to(device) / label_scale

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs_scaled, used_dual = _forward_model(model, batch_data, device)
                loss = loss_function(outputs_scaled, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            if used_dual:
                used_slab_in_epoch += 1

        epoch_loss /= max(1, step)
        epoch_loss_values.append(epoch_loss)

        # Scheduler（兼容你原本的两种）
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                # 自定义 CosineScheduler（返回 lr）
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Logging
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        frac_dual = used_slab_in_epoch / max(1, batch_count_in_epoch)
        writer.add_scalar('Train/FracDualBatches', frac_dual, epoch)

        print(f"epoch {epoch + 1} loss: {epoch_loss:.4f}, lr: {current_lr:.8f}, "
              f"dual_used(frac): {frac_dual:.2f}")

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            rmse_metric.reset()
            mae_metric.reset()

            used_slab_in_val = 0
            val_batch_count = 0

            with torch.no_grad():
                for val_data in valid_loader:
                    val_batch_count += 1
                    val_outputs_scaled, used_dual_val = _forward_model(model, val_data, device)
                    if used_dual_val:
                        used_slab_in_val += 1

                    # 还原到原标注尺度后再算指标
                    val_outputs = val_outputs_scaled * label_scale
                    val_labels = val_data["label"].to(device)

                    # 更新指标
                    rmse_metric(val_outputs, val_labels)
                    mae_metric(val_outputs, val_labels)

                mean_rmse = rmse_metric.aggregate().item()
                mean_mae = mae_metric.aggregate().item()
                rmse_metric.reset()
                mae_metric.reset()

            # Logging & save
            writer.add_scalar('Validation RMSE', mean_rmse, epoch)
            writer.add_scalar('Validation MAE', mean_mae, epoch)
            writer.add_scalar('Val/FracDualBatches',
                              used_slab_in_val / max(1, val_batch_count), epoch)

            print(f'Epoch {epoch + 1}, Val RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}')

            if mean_rmse < best_metric:
                best_metric = mean_rmse
                best_metric_epoch = epoch + 1
                best_metric_mae = mean_mae
                best_ckpt_path = save_model(model, save_dir, epoch + 1, mean_rmse)
                print(f"New best @ epoch {epoch + 1}: RMSE {mean_rmse:.4f}  (ckpt saved)")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stop: RMSE did not improve for {patience} epochs.")
                    break

    writer.close()
    print(f"Training done. Best RMSE: {best_metric:.4f} @ epoch {best_metric_epoch}, "
          f"MAE: {best_metric_mae:.4f}")

    return best_ckpt_path


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
    vit_pool='cls', train_scope='head', fusion='concat'
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

    visualize_training_data(loader=train_loader, num_images=8, mode='triplet', save_path=os.path.join(visual_dir, 'train_samples.png'))
    print('set dataloader...')

    # 3) 模型
    num_classes = 52  
    
    if backbone == 'dinov3':
        model = DualDinoV3LateFusion52(
            rnflt_model_name=dinov3_model,
            rnflt_vit_pool=vit_pool,
            slab_model_name=dinov3_model,
            slab_vit_pool=vit_pool,
            fusion=fusion,               
            head_hidden_dim=512,
            head_dropout=0.1,
            out_dim=num_classes,
        )
        set_train_scope(model, train_scope)

        vis_model = DualDinoV3LateFusion52(
            rnflt_model_name=dinov3_model,
            rnflt_vit_pool=vit_pool,
            slab_model_name=dinov3_model,
            slab_vit_pool=vit_pool,
            fusion=fusion,               
            head_hidden_dim=512,
            head_dropout=0.1,
            out_dim=num_classes,
        ) 
        
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    
    model = model.to(device)
    print('set model...')
    info_summary(model, input_size=[(1, 3, 224, 224), (1, 3, 224, 224)], col_names=["input_size", "output_size", "num_params", "trainable"], verbose=1)
    model = nn.DataParallel(model)

    # 4) 优化器 & 调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    # 5) 训练
    best_ckpt_path = train_model_multi(train_loader, val_loader, model, optimizer, max_epochs,
                                          scheduler, val_interval, device, save_dir, log_dir, patience)
    
    # 6) 可视化
    print("="*30 + " Visualization " + "="*30)
    visualize_after_training_dual(
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