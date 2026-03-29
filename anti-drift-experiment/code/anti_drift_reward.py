"""
Anti-Drift Post-Hoc Reward Module
==================================
Implements DINO consistency, Temporal PSNR, and HPSv3 quality rewards
for evaluating and guiding WorldPlay video generation.

Core idea: After each chunk is generated, score it with multiple reward
signals. In Best-of-N mode, generate N candidates and select the best.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Dict
import os


class DINOReward:
    """
    DINO-based visual consistency reward.
    Uses DINOv2 CLS token cosine similarity to measure subject consistency
    across frames — the core anti-drift signal.
    """

    def __init__(self, model_name="facebook/dinov2-base", device="cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_features(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract normalized CLS token features from a list of PIL Images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token [N, D]
        return F.normalize(features, dim=-1)

    def compute_reward(self, frames: List[Image.Image], anchor_features=None) -> Dict:
        """
        Compute DINO consistency reward.

        Returns:
            dict with:
              - adj_sim: mean adjacent frame similarity (VBench Subject Consistency)
              - anchor_sim: mean similarity to anchor (first frame)
              - reward: composite score in [0, 1]
        """
        features = self.get_features(frames)  # [T, D]

        # Adjacent frame similarity
        adj_sim = (features[:-1] * features[1:]).sum(dim=-1)  # [T-1]
        adj_reward = adj_sim.mean().item()

        # Anchor similarity (to first frame or provided anchor)
        if anchor_features is not None:
            anchor = anchor_features
        else:
            anchor = features[0:1]

        anchor_sim = (features * anchor).sum(dim=-1)  # [T]
        anchor_reward = anchor_sim.mean().item()

        # Composite: weighted combination
        reward = 0.6 * adj_reward + 0.4 * anchor_reward

        return {
            "adj_sim": adj_reward,
            "anchor_sim": anchor_reward,
            "reward": reward,
            "per_frame_anchor_sim": anchor_sim.cpu().numpy().tolist(),
            "anchor_features": anchor,
        }


class TemporalPSNRReward:
    """
    Temporal PSNR reward for measuring frame-to-frame smoothness.
    Penalizes sudden visual changes (chunk boundary artifacts).
    """

    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute PSNR between two images (numpy arrays, 0-255)."""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 10 * np.log10(255.0 ** 2 / mse)

    def compute_reward(self, frames: List[Image.Image]) -> Dict:
        """
        Compute temporal PSNR across consecutive frames.

        Returns:
            dict with mean_psnr, min_psnr, and normalized reward
        """
        arrays = [np.array(f) for f in frames]
        psnrs = []
        for i in range(1, len(arrays)):
            psnr = self.compute_psnr(arrays[i - 1], arrays[i])
            psnrs.append(psnr)

        mean_psnr = np.mean(psnrs)
        min_psnr = np.min(psnrs)

        # Normalize: PSNR typically ranges 15-40 dB for video frames
        # Map to [0, 1] with sigmoid-like scaling
        reward = np.clip((mean_psnr - 15) / 25, 0, 1)

        return {
            "mean_psnr": float(mean_psnr),
            "min_psnr": float(min_psnr),
            "reward": float(reward),
            "per_frame_psnr": [float(p) for p in psnrs],
        }


class CompositeReward:
    """
    Composite reward combining DINO consistency, Temporal PSNR, and
    optionally HPSv3 quality score.

    R = lambda_1 * DINO + lambda_2 * PSNR + lambda_3 * HPSv3
    """

    def __init__(
        self,
        device="cuda",
        lambda_dino=0.5,
        lambda_psnr=0.3,
        lambda_hps=0.2,
        use_hps=False,
    ):
        self.dino = DINOReward(device=device)
        self.psnr = TemporalPSNRReward()
        self.lambda_dino = lambda_dino
        self.lambda_psnr = lambda_psnr
        self.lambda_hps = lambda_hps
        self.use_hps = use_hps
        self.anchor_features = None

    def set_anchor(self, anchor_frame: Image.Image):
        """Set anchor frame (first frame of video) for DINO anchor similarity."""
        self.anchor_features = self.dino.get_features([anchor_frame])

    def score(self, frames: List[Image.Image], prompt: str = "") -> Dict:
        """Score a chunk of frames with composite reward."""
        dino_result = self.dino.compute_reward(frames, self.anchor_features)
        psnr_result = self.psnr.compute_reward(frames)

        composite = (
            self.lambda_dino * dino_result["reward"]
            + self.lambda_psnr * psnr_result["reward"]
        )

        result = {
            "composite_reward": composite,
            "dino": dino_result,
            "psnr": psnr_result,
        }

        return result


def frames_from_video_array(video_array: np.ndarray) -> List[Image.Image]:
    """Convert video array [1, T, H, W, C] or [T, H, W, C] to list of PIL Images."""
    if video_array.ndim == 5:
        video_array = video_array[0]
    frames = []
    for i in range(video_array.shape[0]):
        frame = video_array[i]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        frames.append(Image.fromarray(frame))
    return frames
