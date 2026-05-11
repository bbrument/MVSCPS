"""Load functions for LUCES-MV dataset.

Registered via @REG.register so they are available in Hydra configs.
Imported by load_fn.py to trigger registration.
"""

import json

import cv2
import numpy as np

from core.registry import REG

# Sequential view index → actual view directory number
VIEW_DIRS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66]


@REG.register("fn", name="lucesmv_img")
def load_img_lucesmv(img_fpath, img_downscale=1):
    """Load LUCES-MV 16-bit PNG image.

    img_fpath is constructed by dataset_train.py as:
      {root_dir}/image/{view_light_index}.png
    But LUCES-MV uses a different directory structure, so we parse the
    view-light index and construct the correct path.

    Actually, dataset_train.py constructs the path from the config.
    For LUCES-MV, we store the root_dir as the preprocessed directory,
    and the image path is:
      {root_dir}/{Object}/view_{VIEW_DIRS[v]}/{l+1:02d}.png

    However, the actual path passed by dataset_train is already the full path.
    We just need to read the 16-bit PNG.
    """
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


@REG.register("fn", name="lucesmv_mask")
def load_mask_lucesmv(mask_fpath, img_downscale=1):
    """Load LUCES-MV mask (8-bit binary, 0/255)."""
    mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_fpath}")
    if img_downscale != 1:
        target_size = (int(mask.shape[1] / img_downscale), int(mask.shape[0] / img_downscale))
        mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    return mask


@REG.register("fn", name="lucesmv_normal")
def load_normal_lucesmv(normal_fpath):
    """Load LUCES-MV normals (16-bit RGB PNG, decoded as (pixel/65535)*2 - 1)."""
    normal_img = cv2.imread(normal_fpath, cv2.IMREAD_UNCHANGED)
    if normal_img is None:
        raise FileNotFoundError(f"Normal not found: {normal_fpath}")
    normal_img = normal_img[..., ::-1].astype(np.float32)  # BGR → RGB
    normal_img = (normal_img / 65535.0) * 2.0 - 1.0
    normal_img /= (np.linalg.norm(normal_img, axis=-1, keepdims=True) + 1e-6)
    return normal_img


@REG.register("fn", name="lucesmv_camera")
def load_cameras_lucesmv(cam_fpath):
    """Load camera_params.json created by preprocessing."""
    with open(cam_fpath) as f:
        return json.load(f)


@REG.register("fn", name="lucesmv_camera_select")
def select_camera_lucesmv(view_light_index, cams, img_downscale=1):
    """Extract C2W and K for a given view-light index."""
    C2W = np.array(cams[f'C2W_{view_light_index}'])[:3]  # (3, 4)
    K = np.array(cams[f'K_{view_light_index}'])[:3, :3]   # (3, 3)
    K[:2] /= img_downscale
    return C2W, K


@REG.register("fn", name="lucesmv_light")
def load_lights_lucesmv(lights_fpath):
    """Load light intensities for LUCES-MV.

    Returns intensities as (N_lights, 3) array.
    Note: LUCES-MV has no GT light directions (point light),
    but we return the intensities for the rendering pipeline.
    """
    return np.loadtxt(lights_fpath)
