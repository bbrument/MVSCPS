"""Load functions for IDR format (cameras.npz with world_mat/scale_mat).

Registered via @REG.register so they are available in Hydra configs.
Imported by load_fn.py to trigger registration.
"""

import json
import logging
import os

import cv2
import numpy as np

from core.registry import REG

logger = logging.getLogger(__name__)


def load_K_Rt_from_P(P):
    """Decompose projection matrix P into K (intrinsics) and C2W (camera-to-world).

    Ported from eval_pipeline/src/core/camera.py.

    Args:
        P: (3, 4) or (4, 4) projection matrix

    Returns:
        K: (3, 3) intrinsics matrix
        C2W: (4, 4) camera-to-world pose
    """
    if P.shape[0] == 4:
        P = P[:3, :4]

    K_3x3, R, t, *_ = cv2.decomposeProjectionMatrix(P)
    K_3x3 = K_3x3 / K_3x3[2, 2]

    # Sign check: cv2 may produce improper rotations
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # Camera center in world coordinates
    C = (t[:3] / t[3])[:, 0]

    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = R.T
    C2W[:3, 3] = C

    # Validation: recompose P and check
    P_recomp = K_3x3 @ np.hstack([R, -R @ C.reshape(3, 1)])
    P_norm = P / np.linalg.norm(P)
    P_recomp_norm = P_recomp / np.linalg.norm(P_recomp)
    max_diff = np.max(np.abs(P_norm - P_recomp_norm))
    if max_diff > 1e-4:
        logger.warning(f"P recomposition mismatch: max_diff={max_diff:.6f}")

    return K_3x3, C2W


@REG.register("fn", name="idr_img")
def load_img_idr(img_fpath, img_downscale=1):
    """Load image from IDR format (image/ directory)."""
    img = cv2.imread(img_fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_fpath}")
    img = img[..., ::-1]  # BGR → RGB
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    if img_downscale != 1:
        scale = 1.0 / float(img_downscale)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


@REG.register("fn", name="idr_mask")
def load_mask_idr(mask_fpath, img_downscale=1):
    """Load mask from IDR format (mask/ directory)."""
    mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_fpath}")
    if img_downscale != 1:
        target_size = (int(mask.shape[1] / img_downscale), int(mask.shape[0] / img_downscale))
        mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    return mask


@REG.register("fn", name="idr_camera")
def load_cameras_idr(cam_fpath):
    """Load cameras from IDR format (cameras.npz).

    Converts to the same dict format used by DiligentMV's camera_params.json:
    - O2W_scale, O2W_translation
    - K_{view_light_index}, C2W_{view_light_index}

    Args:
        cam_fpath: path to cameras.npz file

    Returns:
        dict compatible with the MVSCPS camera format
    """
    data = np.load(cam_fpath)

    # Count views
    n_views = len([k for k in data.keys() if k.startswith('world_mat_')])
    logger.info(f"IDR cameras: {n_views} views found")

    cam_dict = {}
    C2W_list = []

    # Extract O2W from scale_mat (NeuS-to-world transform).
    # scale_mat maps NeuS-space → world mm: X_mm = s * X_neus + offset
    # So: O2W_scale = s, O2W_translation = offset
    # Then: obj_space = (mm - offset) / s = NeuS-space (object in unit sphere)
    # And mesh export: v_mm = v_obj * s + offset (back to mm, matches GT)
    S0 = np.array(data.get('scale_mat_0', np.eye(4)), dtype=np.float64)
    s_scale = S0[0, 0]
    offset = S0[:3, 3]

    for i in range(n_views):
        P = np.array(data[f'world_mat_{i}'], dtype=np.float64)  # 4x4 world-to-image

        # Decompose world_mat directly: C2W with camera center in world mm.
        # K and R are identical to P@S decomposition (isotropic scale cancels
        # in K normalization, rotation is unchanged). Only camera center differs:
        # C_mm here vs C_neus = (C_mm - offset) / s from P@S.
        K_3x3, C2W = load_K_Rt_from_P(P[:3, :4])

        C2W_list.append(C2W)

        cam_key = f"V{i:02d}L00"
        cam_dict[f"K_{cam_key}"] = K_3x3.tolist()
        cam_dict[f"C2W_{cam_key}"] = C2W.tolist()

    cam_dict['O2W_scale'] = float(s_scale)
    cam_dict['O2W_translation'] = offset.tolist()

    return cam_dict


@REG.register("fn", name="idr_camera_select")
def select_camera_idr(view_light_index, cams, img_downscale=1):
    """Extract C2W and K for a given view-light index (IDR format).

    Supports multi-light: all lights in a view share the same camera,
    so V04L03 falls back to V04L00 if the specific key is not found.
    """
    key = view_light_index
    if f'C2W_{key}' not in cams:
        # Multi-light fallback: use L00 for the same view
        view_part = key.split('L')[0]
        key = f'{view_part}L00'
    C2W = np.array(cams[f'C2W_{key}'])[:3]  # (3, 4)
    K = np.array(cams[f'K_{key}'])[:3, :3]   # (3, 3)
    K[:2] /= img_downscale
    return C2W, K


@REG.register("fn", name="idr_light")
def load_lights_idr(lights_fpath):
    """Load lights for IDR format — no GT light info available."""
    return None
