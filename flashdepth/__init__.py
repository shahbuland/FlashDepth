"""
FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

This module provides a clean nn.Module interface for using FlashDepth
in external training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .model import FlashDepth as _FlashDepthBase


class FlashDepthModel(nn.Module):
    """
    nn.Module wrapper for FlashDepth with normalized inputs/outputs.

    This class provides a simple interface for using FlashDepth in training pipelines:
    - Accepts images in [-1, 1] range
    - Returns depth maps normalized to [-1, 1] range
    - Can be compiled with torch.compile
    - Can be converted to bfloat16

    Example:
        >>> model = FlashDepthModel(model_size='vits')
        >>> model = model.to('cuda').to(torch.bfloat16)
        >>> model = torch.compile(model, mode='max-autotune')
        >>>
        >>> # Input: (B, 3, H, W) in [-1, 1]
        >>> images = torch.randn(2, 3, 518, 518).cuda().to(torch.bfloat16)
        >>> images = images * 2 - 1  # Scale to [-1, 1]
        >>>
        >>> # Output: (B, H, W) in [-1, 1]
        >>> depth = model(images)
    """

    def __init__(
        self,
        model_size: str = 'vits',
        use_mamba: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize FlashDepth model.

        Args:
            model_size: Size of ViT encoder ('vits' or 'vitl')
            use_mamba: Whether to use Mamba temporal module
            checkpoint_path: Optional path to checkpoint. If None, model is randomly initialized.
        """
        super().__init__()

        self.model_size = model_size
        self.use_mamba = use_mamba

        # Initialize base FlashDepth model
        self.model = _FlashDepthBase(
            vit_size=model_size,
            use_mamba=use_mamba,
            training=False,  # Inference mode
            batch_size=1,
            # Mamba configs (only used if use_mamba=True)
            mamba_type="add",
            num_mamba_layers=4,
            downsample_mamba=[0.1],
            mamba_in_dpt_layer=[1],
            mamba_d_conv=4,
            mamba_d_state=256,
            use_hydra=False,
            use_transformer_rnn=False,
            use_xlstm=False,
            hybrid_configs=None,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # ImageNet normalization stats (for input preprocessing)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Depth normalization stats (computed on first forward pass)
        self.depth_min = None
        self.depth_max = None

        # Create compilable inference function (like in single_frame_pipeline.py)
        def _inference_fn(frame, patch_h, patch_w):
            """Compilable inference path without control flow."""
            B, C, H, W = frame.shape
            dpt_features = self.model.get_dpt_features(frame, input_shape=(B, C, H, W))
            depth_pred = self.model.final_head(dpt_features, patch_h, patch_w)
            return torch.clip(depth_pred, min=0)

        self._inference_fn = _inference_fn

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)

    def compile(self, mode='max-autotune', **kwargs):
        """
        Compile the inference function for faster execution.

        Args:
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            **kwargs: Additional arguments for torch.compile

        Returns:
            self (for chaining)
        """
        self._inference_fn = torch.compile(self._inference_fn, mode=mode, **kwargs)
        return self

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input from [-1, 1] to ImageNet normalized [0, 1].

        Args:
            x: Input tensor (B, 3, H, W) in [-1, 1] range

        Returns:
            Preprocessed tensor with ImageNet normalization
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std

        return x

    def postprocess_output(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Normalize depth output to [-1, 1] range.

        Args:
            depth: Raw depth prediction (B, H, W)

        Returns:
            Normalized depth in [-1, 1] range
        """
        # Use running min/max for stable normalization
        if self.training:
            batch_min = depth.min().detach()
            batch_max = depth.max().detach()

            if self.depth_min is None:
                self.depth_min = batch_min
                self.depth_max = batch_max
            else:
                # Exponential moving average
                momentum = 0.1
                self.depth_min = (1 - momentum) * self.depth_min + momentum * batch_min
                self.depth_max = (1 - momentum) * self.depth_max + momentum * batch_max
        else:
            # Use batch statistics during inference
            batch_min = depth.min().detach()
            batch_max = depth.max().detach()

            if self.depth_min is None or self.depth_max is None:
                self.depth_min = batch_min
                self.depth_max = batch_max

        # Normalize to [0, 1]
        depth_range = self.depth_max - self.depth_min
        if depth_range > 0:
            depth_normalized = (depth - self.depth_min) / depth_range
        else:
            depth_normalized = torch.zeros_like(depth)

        # Convert to [-1, 1]
        depth_normalized = depth_normalized * 2.0 - 1.0

        return depth_normalized

    def _resize_to_multiple_of(self, size: int, multiple: int) -> int:
        """Round size to nearest multiple."""
        return int(round(size / multiple) * multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic resizing to multiples of patch_size.

        Args:
            x: Input images (B, 3, H, W) in [-1, 1] range

        Returns:
            Depth maps (B, H, W) in [-1, 1] range, same spatial size as input
        """
        B, C, H_orig, W_orig = x.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # Calculate target size (nearest multiple of patch_size)
        patch_size = self.model.patch_size
        H_model = max(patch_size, self._resize_to_multiple_of(H_orig, patch_size))
        W_model = max(patch_size, self._resize_to_multiple_of(W_orig, patch_size))

        # Resize input if needed
        needs_resize = (H_orig != H_model) or (W_orig != W_model)
        if needs_resize:
            x = F.interpolate(
                x,
                size=(H_model, W_model),
                mode='bilinear',
                align_corners=False
            )

        # Preprocess input
        x = self.preprocess_input(x)

        # Run inference using compiled function
        patch_h, patch_w = H_model // patch_size, W_model // patch_size
        depth = self._inference_fn(x, patch_h, patch_w)

        # Remove channel dimension: (B, 1, H, W) -> (B, H, W)
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        # Resize back to original size if needed
        if needs_resize:
            depth = F.interpolate(
                depth.unsqueeze(1),  # Add channel dim for interpolation
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Remove channel dim

        # Postprocess output to [-1, 1]
        depth = self.postprocess_output(depth)

        return depth


# Convenience exports
__all__ = ['FlashDepthModel', 'FlashDepth']

# Also export the base model for advanced users
FlashDepth = _FlashDepthBase
