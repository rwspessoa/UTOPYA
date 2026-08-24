"""
Image cache builder for UTOPYA (Section 5.9's Image modality, A14
frozen-backbone extension).

Wires the "12_..._Image" per-experiment camera videos (Cam0.mp4, Cam1.mp4,
Cam2.mp4 — three fixed camera views of the plant) into a per-experiment
fixed-size 512-d visual summary that ImageEncoder (src/models/encoders.py)
can consume: one representative frame is pulled from each available camera,
run through a frozen ImageNet-pretrained ResNet-18 (paper's framing of
ResNet-18 as "strong low-level feature extraction" from a backbone that
isn't fine-tuned), and the per-camera 512-d features are mean-pooled across
however many of the 3 cameras were actually available for that experiment
(matching the paper's Eq. for image mean-pooling across cameras).

Precomputing these features once at cache-build time (rather than decoding
video and running the CNN inside the training loop every epoch) is far
cheaper computationally and mirrors how TextEncoder consumes a precomputed
SBERT embedding rather than running the language model live. Treated as an
experiment-level summary (constant across all windows), the same
simplification already used for Audio.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_ROOT_DIRNAME = "12_Batch_Distillation_Plant_M-202210_Image"

N_IMAGE_FEATURES = 512
CAMERA_NAMES = ["Cam0", "Cam1", "Cam2"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Lazily-constructed module-level singleton: build the frozen ResNet-18
# feature extractor once and reuse it across all experiments (avoids
# re-loading pretrained weights for every experiment).
_resnet_extractor = None


def _get_resnet_extractor(device: str = "cpu"):
    global _resnet_extractor
    if _resnet_extractor is None:
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # drop final FC, keep (B,512,1,1) pooled output
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        _resnet_extractor = backbone.to(device)
    return _resnet_extractor


def _extract_frame(video_path: str) -> Optional[np.ndarray]:
    """Read the middle frame of a video as a (224, 224, 3) float32 RGB array in [0, 1]."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    frame = frame[:, :, ::-1]                       # BGR → RGB
    frame = cv2.resize(frame, (224, 224))
    return frame.astype(np.float32) / 255.0


def _load_experiment_image_features(
    data_root: str, system: str, op: str, name: str, device: str = "cpu",
) -> Optional[np.ndarray]:
    """Return (512,) mean-pooled ResNet-18 feature vector, or None if no camera video usable."""
    frames = []
    for cam in CAMERA_NAMES:
        video_path = os.path.join(
            data_root, IMAGE_ROOT_DIRNAME, "Operation", system, op, name, f"{cam}.mp4"
        )
        if not os.path.exists(video_path):
            continue
        frame = _extract_frame(video_path)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    import torch

    batch = np.stack(frames, axis=0)                          # (n_frames, 224, 224, 3)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    batch = (batch - mean) / std
    batch = batch.transpose(0, 3, 1, 2)                        # (n_frames, 3, 224, 224) NCHW

    tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
    extractor = _get_resnet_extractor(device)
    with torch.no_grad():
        feats = extractor(tensor)                              # (n_frames, 512, 1, 1)
    feats = feats.squeeze(-1).squeeze(-1)                       # (n_frames, 512)
    pooled = feats.mean(dim=0)                                  # mean-pool across cameras → (512,)
    return pooled.cpu().numpy().astype(np.float32)


def build_image_feature_cache(
    data_root: str,
    system: str,
    experiments: List[Dict],   # dicts with keys "op" and "name"
    device: str = "cpu",
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build per-experiment ResNet-18 visual summaries.

    Returns dict (op, name) → np.ndarray (N_IMAGE_FEATURES,).
    Experiments with no usable camera video fall back to an all-zero array
    (same convention as the tabular/molecular/audio caches).
    """
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing = 0
    for meta in experiments:
        op, name = meta["op"], meta["name"]
        feat = _load_experiment_image_features(data_root, system, op, name, device=device)
        if feat is None:
            feat = np.zeros(N_IMAGE_FEATURES, dtype=np.float32)
            missing += 1
        cache[(op, name)] = feat
    if missing:
        print(f"[Image] {missing}/{len(experiments)} experiments had no usable camera video — zero-filled fallback.")
    print(f"[Image] Built ResNet-18 feature cache for {len(cache)} experiments ({N_IMAGE_FEATURES}-d).")
    return cache
