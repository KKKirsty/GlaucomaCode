import torch
import torch.nn as nn
from transformers import AutoModel


# ======================== rnflt + slab ======================== #






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
