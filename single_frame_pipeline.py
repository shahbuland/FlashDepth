"""
Simple single-frame inference pipeline for FlashDepth.

This provides a clean API for running depth estimation on individual images
without needing to understand the full training infrastructure.

Usage:
    from single_frame_pipeline import FlashDepthPipeline

    # Initialize pipeline
    pipeline = FlashDepthPipeline(
        checkpoint_path="path/to/checkpoint.pth",
        model_size="vits",  # or "vitl"
        use_mamba=False,    # Set True for temporal coherence across frames
        device="cuda"
    )

    # Run inference on single image
    depth_map = pipeline.predict("path/to/image.jpg")

    # Or process a batch of images
    depths = pipeline.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

    # Save depth visualization
    pipeline.save_depth_visualization(depth_map, "output.png")
"""

import os
import logging
from typing import Union, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from flashdepth.model import FlashDepth
from dataloaders.depthanything_preprocess import depthanything_preprocess


class FlashDepthPipeline:
    """Simple pipeline for single-frame depth estimation with FlashDepth."""

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "vits",
        use_mamba: bool = False,
        device: str = "cuda",
        compile_model: bool = False,
    ):
        """
        Initialize the FlashDepth inference pipeline.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            model_size: Size of Vision Transformer encoder ("vits" or "vitl")
            use_mamba: Whether to use Mamba temporal module (for video sequences)
            device: Device to run inference on ("cuda" or "cpu")
            compile_model: Whether to use torch.compile for speedup (requires PyTorch 2.0+)
        """
        self.device = device
        self.use_mamba = use_mamba
        self.model_size = model_size

        # Initialize model
        logging.info(f"Initializing FlashDepth with {model_size} encoder...")
        self.model = FlashDepth(
            vit_size=model_size,
            use_mamba=use_mamba,
            training=False,
            batch_size=1,
            # Mamba-specific configs (only used if use_mamba=True)
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

        # Load checkpoint
        logging.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        # Optionally compile model for faster inference
        if compile_model:
            logging.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        logging.info("Pipeline initialized successfully!")

    def preprocess_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Load and preprocess an image for inference.

        Args:
            image: Path to image file, numpy array (H,W,C in RGB), or PIL Image
            target_size: Optional (width, height) to resize to. If None, maintains aspect ratio
                        with dimensions as multiples of 14

        Returns:
            Preprocessed image tensor of shape (1, 3, H, W)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure numpy array in float32 [0, 1] range
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Resize if target size specified
        if target_size is not None:
            target_w, target_h = target_size
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            image = np.clip(image, 0, 1).astype(np.float32)

        # Apply DepthAnything preprocessing (resize to multiple of 14, normalize)
        h, w = image.shape[:2]
        image_tensor = depthanything_preprocess(
            image,
            width=w if target_size is None else target_size[0],
            height=h if target_size is None else target_size[1],
            to_tensor=True,
            color_aug=False
        )

        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        # Ensure float32 dtype
        image_tensor = image_tensor.float().unsqueeze(0)

        return image_tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Run depth estimation on a single image.

        Args:
            image: Path to image file, numpy array (H,W,C), or PIL Image
            target_size: Optional (width, height) to resize to
            return_numpy: If True, returns numpy array. If False, returns torch tensor

        Returns:
            Depth map of shape (H, W) with depth values
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image, target_size)
        image_tensor = image_tensor.to(self.device)

        # Add temporal dimension for model: (1, 3, H, W) -> (1, 1, 3, H, W)
        image_tensor = image_tensor.unsqueeze(1)

        # Reset Mamba state if using temporal module
        if self.use_mamba:
            self.model.mamba.start_new_sequence()

        # Run inference
        # Model expects: (B, T, C, H, W) and returns loss_dict, grid
        # For single frame inference, we just need the depth prediction
        B, T, C, H, W = image_tensor.shape

        # Process through model
        frame = image_tensor[:, 0, :, :, :]  # (1, 3, H, W)
        patch_h, patch_w = H // self.model.patch_size, W // self.model.patch_size

        # Get DPT features
        dpt_features = self.model.get_dpt_features(frame, input_shape=(B, C, H, W))

        # Get depth prediction
        depth_pred = self.model.final_head(dpt_features, patch_h, patch_w)
        depth_pred = torch.clip(depth_pred, min=0)

        # Remove batch dimension: (1, H, W) -> (H, W)
        depth_pred = depth_pred.squeeze(0)

        if return_numpy:
            return depth_pred.cpu().numpy()
        else:
            return depth_pred

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        target_size: Optional[Tuple[int, int]] = None,
        return_numpy: bool = True,
    ) -> Union[List[np.ndarray], List[torch.Tensor]]:
        """
        Run depth estimation on a batch of images.

        Args:
            images: List of image paths, numpy arrays, or PIL Images
            target_size: Optional (width, height) to resize to
            return_numpy: If True, returns list of numpy arrays

        Returns:
            List of depth maps, one per input image
        """
        depths = []
        for img in images:
            depth = self.predict(img, target_size, return_numpy)
            depths.append(depth)
        return depths

    def save_depth_visualization(
        self,
        depth_map: Union[np.ndarray, torch.Tensor],
        output_path: str,
        colormap: str = "inferno",
        dpi: int = 100,
    ):
        """
        Save depth map as a colored visualization.

        Args:
            depth_map: Depth map array of shape (H, W)
            output_path: Path to save visualization
            colormap: Matplotlib colormap name (e.g., "inferno", "viridis", "magma")
            dpi: DPI for saved image
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        # Normalize to [0, 1] for visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth_map

        # Apply colormap
        plt.figure(figsize=(depth_map.shape[1] / dpi, depth_map.shape[0] / dpi), dpi=dpi)
        plt.imshow(depth_normalized, cmap=colormap)
        plt.axis("off")
        plt.tight_layout(pad=0)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close()

        logging.info(f"Saved depth visualization to {output_path}")

    def save_depth_npy(
        self,
        depth_map: Union[np.ndarray, torch.Tensor],
        output_path: str,
    ):
        """
        Save depth map as numpy array.

        Args:
            depth_map: Depth map array of shape (H, W)
            output_path: Path to save .npy file
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, depth_map)

        logging.info(f"Saved depth array to {output_path}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example: Initialize pipeline
    pipeline = FlashDepthPipeline(
        checkpoint_path="checkpoints/flashdepth_vitl.pth",
        model_size="vitl",
        use_mamba=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=False,
    )

    # Example: Single image inference
    depth = pipeline.predict("examples/example_image.jpg")
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Example: Save visualization
    pipeline.save_depth_visualization(depth, "output/depth_vis.png")

    # Example: Save raw depth
    pipeline.save_depth_npy(depth, "output/depth.npy")

    # Example: Batch inference
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    depths = pipeline.predict_batch(image_paths)
    print(f"Processed {len(depths)} images")
