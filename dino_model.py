import torch
import torch.nn as nn
from transformers import AutoModel


# ======================== rnflt + slab ======================== #
# =======================
# 单骨干：抽取 [B, D] 特征
# =======================
class DinoV3FeatureBackbone(nn.Module):
    """
    把 HF DINOv3 (ConvNeXt/ViT) 封装为：输入 [B,3,H,W] -> 输出 [B, D] 全局特征
    与DinoV3Backbone52类一致，只是把回归头移除，专注于特征抽取。
    """
    def __init__(
        self,
        hf_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        vit_pool: str = "cls",          # "cls" | "mean_patch" | "pooler"
        apply_imagenet_norm: bool = False,
        freeze_backbone: bool = False,  # 可选：初始化时就冻结
    ):
        super().__init__()
        self.net = AutoModel.from_pretrained(hf_model_name)
        self.apply_imagenet_norm = apply_imagenet_norm
        self.vit_pool = vit_pool

        # 判别分支
        self.is_convnext = hasattr(self.net.config, "hidden_sizes") and isinstance(
            self.net.config.hidden_sizes, (list, tuple)
        )

        # ViT 相关
        self.hidden_dim = getattr(self.net.config, "hidden_size", None)
        self.num_register_tokens = int(getattr(self.net.config, "num_register_tokens", 0))

        # ConvNeXt 最后通道数
        if self.is_convnext:
            self.out_dim = self.net.config.hidden_sizes[-1]
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            assert self.hidden_dim is not None, "无法从 HF ViT config 读取 hidden_size"
            self.out_dim = self.hidden_dim

        # 可选的 ImageNet 归一化
        if self.apply_imagenet_norm:
            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if freeze_backbone:
            self.set_trainable(False)

    @torch.no_grad()
    def set_trainable(self, flag: bool = True):
        for p in self.net.parameters():
            p.requires_grad = flag

    def _imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.im_mean.to(device=x.device, dtype=x.dtype)
        std = self.im_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _vit_aggregate(self, tokens: torch.Tensor, pool: str) -> torch.Tensor:
        """
        tokens: [B, 1 + reg + N, D]
        return: [B, D]
        """
        B, T, D = tokens.shape
        reg = self.num_register_tokens
        assert T >= 1 + reg, f"序列长度({T}) 小于 1+register({1+reg})"

        if pool == "cls":
            return tokens[:, 0, :]
        elif pool == "mean_patch":
            start = 1 + reg
            return tokens[:, start:, :].mean(dim=1)
        elif pool == "pooler":
            # 仅当模型提供 pooler_output 时有效，否则退回 CLS
            return tokens[:, 0, :]
        else:
            raise ValueError("vit_pool 必须是 'cls' | 'mean_patch' | 'pooler'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]  可能尚未做 ImageNet 归一化（由 apply_imagenet_norm 决定）
        return: [B, D]
        """
        if self.apply_imagenet_norm:
            x = self._imagenet_norm(x)

        out = self.net(pixel_values=x, output_hidden_states=False, return_dict=True)

        if self.is_convnext:
            # last_hidden_state: [B, C, H', W'] -> GAP -> [B, C]
            feats = out.last_hidden_state
            feats = self.gap(feats).flatten(1)  # [B, C]
        else:
            # ViT: last_hidden_state [B, 1+reg+N, D]
            if self.vit_pool == "pooler" and getattr(out, "pooler_output", None) is not None:
                feats = out.pooler_output  # [B, D]
            else:
                feats = self._vit_aggregate(out.last_hidden_state, self.vit_pool)  # [B, D]

        return feats  # [B, D]


# =======================
# 融合头（晚期融合）+ 52 回归
# =======================
class LateFusionHead52(nn.Module):
    """
    将两路特征 late fusion 后做 52 维回归。
    支持四种融合方式：
      - 'concat'    : [fr | fs]  -> MLP -> 52
      - 'sum'       : fr + fs    -> MLP -> 52   （先线性对齐到同维，再求和）
      - 'gated-sum' : g∈[0,1]，g*fr + (1-g)*fs -> MLP -> 52
      - 'attn'      : 简单双向注意力后聚合（轻量实现），再 MLP -> 52
    """
    def __init__(
        self,
        dim_rnflt: int,
        dim_slab: int,
        out_dim: int = 52,
        fusion: str = "concat",       # 'concat' | 'sum' | 'gated-sum' | 'attn'
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fusion = fusion

        # 统一对齐到同维度（便于 sum / attn 等）
        common_dim = max(dim_rnflt, dim_slab)
        self.proj_r = nn.Identity() if dim_rnflt == common_dim else nn.Linear(dim_rnflt, common_dim)
        self.proj_s = nn.Identity() if dim_slab == common_dim else nn.Linear(dim_slab, common_dim)

        if fusion == "concat":
            in_dim = dim_rnflt + dim_slab
            self.merge = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif fusion == "sum":
            in_dim = common_dim
            self.merge = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif fusion == "gated-sum":
            # 门控 g = sigmoid(MLP([fr|fs]))
            gate_in = dim_rnflt + dim_slab
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            in_dim = common_dim
            self.merge = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif fusion == "attn":
            # 轻量注意力：把 fr, fs 视作 2 个 token，做一次自注意力
            self.q = nn.Linear(common_dim, common_dim, bias=False)
            self.k = nn.Linear(common_dim, common_dim, bias=False)
            self.v = nn.Linear(common_dim, common_dim, bias=False)
            self.attn_ln = nn.LayerNorm(common_dim)
            in_dim = common_dim
            self.merge = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError("fusion 必须是 'concat' | 'sum' | 'gated-sum' | 'attn'")

    def forward(self, fr: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        """
        fr: [B, Dr] 来自 rnflt 的特征
        fs: [B, Ds] 来自 slab  的特征
        return: [B, 52]
        """
        fr_p = self.proj_r(fr)
        fs_p = self.proj_s(fs)

        if self.fusion == "concat":
            x = torch.cat([fr, fs], dim=1)
            return self.merge(x)

        elif self.fusion == "sum":
            x = fr_p + fs_p
            return self.merge(x)

        elif self.fusion == "gated-sum":
            g = self.gate(torch.cat([fr, fs], dim=1))          # [B,1]
            x = g * fr_p + (1.0 - g) * fs_p                    # [B,C]
            return self.merge(x)

        elif self.fusion == "attn":
            # tokens: [B, 2, C]
            tokens = torch.stack([fr_p, fs_p], dim=1)
            Q = self.q(tokens)  # [B,2,C]
            K = self.k(tokens)
            V = self.v(tokens)
            attn = torch.softmax((Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5), dim=-1)  # [B,2,2]
            fused = (attn @ V).mean(dim=1)  # [B,C]  简单平均两 token 的输出
            fused = self.attn_ln(fused)
            return self.merge(fused)


# =======================
# 双骨干 晚期融合 总模型
# =======================
class DualDinoV3LateFusion52(nn.Module):
    """
    两个独立的 DINOv3 骨干（各自加载预训练、各自可选是否归一化/冻结），
    晚期融合后做 52 维回归。
    forward 接口：
        model(rnflt=x_r, slab=x_s)  -> [B, 52]
    """
    def __init__(
        self,
        # rnflt 分支
        rnflt_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        rnflt_apply_imagenet_norm: bool = False,
        rnflt_vit_pool: str = "cls",
        rnflt_freeze: bool = False,

        # slab 分支
        slab_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        slab_apply_imagenet_norm: bool = False,
        slab_vit_pool: str = "cls",
        slab_freeze: bool = False,

        # 融合与头
        fusion: str = "concat",           # 'concat' | 'sum' | 'gated-sum' | 'attn'
        head_hidden_dim: int = 512,
        head_dropout: float = 0.0,
        out_dim: int = 52
    ):
        super().__init__()
        # 两个独立骨干
        self.backbone_r = DinoV3FeatureBackbone(
            hf_model_name=rnflt_model_name,
            vit_pool=rnflt_vit_pool,
            apply_imagenet_norm=rnflt_apply_imagenet_norm,
            freeze_backbone=rnflt_freeze,
        )
        self.backbone_s = DinoV3FeatureBackbone(
            hf_model_name=slab_model_name,
            vit_pool=slab_vit_pool,
            apply_imagenet_norm=slab_apply_imagenet_norm,
            freeze_backbone=slab_freeze,
        )

        # 融合 + 回归头
        self.head = LateFusionHead52(
            dim_rnflt=self.backbone_r.out_dim,
            dim_slab=self.backbone_s.out_dim,
            out_dim=out_dim,
            fusion=fusion,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    @torch.no_grad()
    def freeze_backbones(self, rnflt: bool = True, slab: bool = True):
        if rnflt:
            self.backbone_r.set_trainable(False)
        if slab:
            self.backbone_s.set_trainable(False)

    @torch.no_grad()
    def unfreeze_backbones(self, rnflt: bool = True, slab: bool = True):
        if rnflt:
            self.backbone_r.set_trainable(True)
        if slab:
            self.backbone_s.set_trainable(True)

    def forward(self, rnflt: torch.Tensor, slab: torch.Tensor) -> torch.Tensor:
        """
        rnflt: [B,3,H,W]
        slab : [B,3,H,W]
        return: [B,52]
        """
        fr = self.backbone_r(rnflt)  # [B, Dr]
        fs = self.backbone_s(slab)   # [B, Ds]
        y  = self.head(fr, fs)       # [B, 52]
        return y





# ======================== rnflt only ======================== #
class DinoV3Backbone52(nn.Module):
    """
    HF DINOv3 backbone → 52 维回归
    - 兼容 ConvNeXt / ViT
    - 输入: x [B,3,H,W]（已做或未做 ImageNet 归一化，取决于 apply_imagenet_norm）
    - 输出: [B,52]
    """
    def __init__(
        self,
        hf_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        out_dim: int = 52,
        apply_imagenet_norm: bool = False,
        vit_pool: str = "cls",  # "cls" | "mean_patch" | "pooler"
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = AutoModel.from_pretrained(hf_model_name)
        self.apply_imagenet_norm = apply_imagenet_norm
        self.vit_pool = vit_pool

        # 判别分支
        self.is_convnext = hasattr(self.net.config, "hidden_sizes") and isinstance(
            self.net.config.hidden_sizes, (list, tuple)
        )
        # 对 ViT: 读 hidden_size & register 数
        self.hidden_dim = getattr(self.net.config, "hidden_size", None)
        self.num_register_tokens = int(getattr(self.net.config, "num_register_tokens", 0))

        if self.is_convnext:
            C = self.net.config.hidden_sizes[-1]
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(C, out_dim),
            )
        else:
            assert self.hidden_dim is not None, "无法从 HF ViT config 读取 hidden_size"
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_dim, out_dim),
            )

        if self.apply_imagenet_norm:
            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.im_mean.to(device=x.device, dtype=x.dtype)
        std = self.im_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _vit_aggregate(self, tokens: torch.Tensor, pool: str) -> torch.Tensor:
        """
        tokens: [B, 1 + reg + N, D]
        返回: [B, D]
        """
        B, T, D = tokens.shape
        reg = self.num_register_tokens
        assert T >= 1 + reg, f"序列长度({T}) 小于 1+register({1+reg})"

        if pool == "cls":
            return tokens[:, 0, :]
        elif pool == "mean_patch":
            # 丢 CLS + register，仅对 patch tokens 做均值
            start = 1 + reg
            return tokens[:, start:, :].mean(dim=1)
        elif pool == "pooler":
            # 仅当 HF 模型提供 pooler_output 时有效，否则退回 CLS
            return tokens[:, 0, :]
        else:
            raise ValueError("vit_pool 必须是 'cls' | 'mean_patch' | 'pooler'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_imagenet_norm:
            x = self._imagenet_norm(x)

        out = self.net(pixel_values=x, output_hidden_states=False, return_dict=True)

        if self.is_convnext:
            # ConvNeXt: last_hidden_state [B, C, H', W'] → GAP → [B, C]
            feats = out.last_hidden_state
            feats = self.pool(feats).flatten(1)
        else:
            # ViT: last_hidden_state [B, 1+reg+N, D]
            if self.vit_pool == "pooler" and getattr(out, "pooler_output", None) is not None:
                feats = out.pooler_output  # [B, D]，已是 CLS 聚合
            else:
                feats = self._vit_aggregate(out.last_hidden_state, self.vit_pool)

        return self.head(feats)  # [B, 52]
