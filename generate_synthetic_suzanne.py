"""
Generate synthetic Suzanne datasets for MVSCPS.

Produces MVSCPS native format:
    output_dir/
        image/V{VV}L{LL}.png
        mask/V{VV}.png
        camera_params.json
        light_directions.txt | light_positions_cam_scaled.npz
        light_intensities.txt
        view_light_idx_{train,val,test}.txt          # all lights
        view_light_idx_maxgray_{train,val,test}.txt  # 1L/view: max mean gray
        view_light_idx_mingray_{train,val,test}.txt  # 1L/view: min mean gray
        view_light_idx_medgray_{train,val,test}.txt  # 1L/view: closest-to-median gray
        view_light_idx_maxsobel_{train,val,test}.txt # 1L/view: max local contrast (Sobel)
        view_light_idx_1l_front_{train,val,test}.txt # fixed L00 for all views
        view_light_idx_1l_maxvar_{train,val,test}.txt# fixed Lk maximizing cross-view variance
        gt_mesh.ply

Two lighting modes:
    - directional: uniform hemisphere directions in camera space
    - point: LEDs mounted close to camera (LUCES-MV style)

Camera sampling modes:
    - rings: concentric rings at fixed elevations (legacy)
    - fibonacci: full Fibonacci sphere distribution

Usage:
    python generate_synthetic_suzanne.py --light_type point --camera_sampling fibonacci \\
        --n_views 50 --n_lights 50 --colored_albedo --shadows \\
        --output_dir data/synthetic_suzanne_point_colored_shadows_fib50
"""

import argparse
import json
import math
import os
import shutil

import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def look_at(camera_pos, target=np.zeros(3), up=np.array([0, 0, 1])):
    """Camera-to-world in Blender convention (X-right, Y-up, -Z-forward)."""
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = camera_pos
    return mat


def blender_c2w_to_opencv_c2w(c2w_blender):
    """Blender c2w -> OpenCV c2w (flip Y and Z)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_blender @ flip


def generate_camera_positions_fibonacci(n_views, camera_distance):
    """Camera positions uniformly distributed on a full sphere via Fibonacci lattice.

    Uses the golden-angle / sunflower mapping:
        cos(polar) = 1 - (2i+1)/n   → evenly spaced in cos(θ), so equal area bands
        azimuth    = 2π * i / φ      → golden angle for uniform azimuthal coverage

    All cameras look at the origin with Z-up preferred.
    """
    positions = []
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n_views):
        cos_theta = 1.0 - (2.0 * i + 1.0) / n_views   # in (-1, 1), top→bottom
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))
        phi = 2.0 * math.pi * i / golden_ratio
        x = camera_distance * sin_theta * math.cos(phi)
        y = camera_distance * sin_theta * math.sin(phi)
        z = camera_distance * cos_theta
        positions.append(np.array([x, y, z]))
    return positions


def generate_camera_positions(n_views, n_rings, camera_distance):
    """Camera positions on hemisphere rings at elevations -10, 0, +30 deg."""
    elevations = np.linspace(-10, 30, n_rings)
    candidates = []
    for ring_idx, elev_deg in enumerate(elevations):
        elev_rad = math.radians(elev_deg)
        n_per_ring = max(1, round(n_views / n_rings))
        azimuth_offset = ring_idx * math.pi / n_rings
        for j in range(n_per_ring * 3):
            azimuth = 2.0 * math.pi * j / (n_per_ring * 3) + azimuth_offset
            x = camera_distance * math.cos(elev_rad) * math.cos(azimuth)
            y = camera_distance * math.cos(elev_rad) * math.sin(azimuth)
            z = camera_distance * math.sin(elev_rad)
            candidates.append(np.array([x, y, z]))

    if len(candidates) <= n_views:
        return candidates
    indices = np.round(np.linspace(0, len(candidates) - 1, n_views)).astype(int)
    return [candidates[i] for i in indices]


def generate_light_directions_hemisphere(n_lights):
    """Uniform directions on the frontal hemisphere (z > 0 in camera space).

    Returns directions in OpenGL camera space convention (to match the
    OpenGL->OpenCV flip done by MVSCPS lighting loader).
    In OpenGL cam space: X-right, Y-up, -Z-forward.
    Frontal hemisphere = Z < 0 (pointing towards the scene).
    """
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_lights):
        theta = math.acos(1 - (i + 0.5) / n_lights)
        phi = golden_angle * i
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        # OpenGL cam: Y-up, -Z-forward
        # Convention: direction FROM surface TO light source
        # Frontal hemisphere = +Z in OpenGL (pointing behind camera, towards light)
        directions.append(np.array([x, y, z]))
    return np.array(directions)


def generate_led_positions(n_lights, led_distance):
    """LED positions on hemisphere in front of camera (camera-mounted rig).

    Positions in camera space (OpenCV: X-right, Y-down, Z-forward).
    Full hemisphere coverage (theta up to ~85°) for varied shading.
    """
    positions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_lights):
        cos_theta = 1 - (i + 0.5) / n_lights
        sin_theta = math.sqrt(1 - cos_theta ** 2)
        phi = golden_angle * i
        x = led_distance * sin_theta * math.cos(phi)
        y = led_distance * sin_theta * math.sin(phi)
        z = led_distance * cos_theta
        positions.append(np.array([x, y, z]))
    return np.array(positions)


def generate_fixed_point_lights_world(n_lights, radius):
    """Fixed point lights on upper hemisphere (z > 0) in world space.

    Golden-angle distribution for uniform coverage.
    """
    positions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_lights):
        cos_theta = 1 - (i + 0.5) / n_lights
        sin_theta = math.sqrt(1 - cos_theta ** 2)
        phi = golden_angle * i
        x = radius * sin_theta * math.cos(phi)
        y = radius * sin_theta * math.sin(phi)
        z = radius * cos_theta
        positions.append(np.array([x, y, z]))
    return np.array(positions)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def barycentric_interpolate_normals(mesh, ray_locations, tri_indices):
    """Smooth normals via barycentric interpolation."""
    n_hits = len(tri_indices)
    if n_hits == 0:
        return np.zeros((0, 3))
    face_verts = mesh.vertices[mesh.faces[tri_indices]]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    e0 = v1 - v0
    e1 = v2 - v0
    ep = ray_locations - v0
    d00 = np.sum(e0 * e0, axis=1)
    d01 = np.sum(e0 * e1, axis=1)
    d11 = np.sum(e1 * e1, axis=1)
    dp0 = np.sum(ep * e0, axis=1)
    dp1 = np.sum(ep * e1, axis=1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    u = (d11 * dp0 - d01 * dp1) / denom
    v = (d00 * dp1 - d01 * dp0) / denom
    w = 1.0 - u - v
    vertex_normals = mesh.vertex_normals
    fn = mesh.faces[tri_indices]
    n0 = vertex_normals[fn[:, 0]]
    n1 = vertex_normals[fn[:, 1]]
    n2 = vertex_normals[fn[:, 2]]
    normals = w[:, None] * n0 + u[:, None] * n1 + v[:, None] * n2
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return normals / norms


def generate_patchwork_albedo(mesh, n_patches=12, seed=42):
    """Assign random colors per patch (Elmer-style Voronoi on face centroids)."""
    rng = np.random.RandomState(seed)
    centroids = mesh.triangles_center
    seeds = rng.choice(len(centroids), n_patches, replace=False)
    seed_points = centroids[seeds]
    dists = np.linalg.norm(centroids[:, None, :] - seed_points[None, :, :], axis=-1)
    patch_idx = np.argmin(dists, axis=1)
    colors = rng.uniform(0.2, 0.9, (n_patches, 3))
    face_colors = colors[patch_idx]
    return face_colors


def raycast_view(mesh, ray_origins, dirs_world, resolution):
    """Single ray-cast per view. Returns hit info reusable for all lights.

    Returns:
        hit_rays: (N_hits,) indices of rays that hit the mesh
        hit_locs: (N_hits, 3) world-space hit locations
        hit_normals: (N_hits, 3) interpolated normals
        mask: (H, W) float64 binary mask
    """
    H, W = resolution, resolution
    origins_flat = np.broadcast_to(ray_origins, (H * W, 3)).copy()
    dirs_flat = dirs_world.reshape(-1, 3).copy()

    # intersects_location with multiple_hits=False uses embree's closest-hit
    locations, idx_ray, idx_tri = mesh.ray.intersects_location(
        origins_flat, dirs_flat, multiple_hits=False
    )

    mask = np.zeros(H * W, dtype=np.float64)
    if len(idx_ray) == 0:
        return np.array([], dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 3)), np.array([], dtype=np.int64), mask.reshape(H, W)

    normals = barycentric_interpolate_normals(mesh, locations, idx_tri)
    mask[idx_ray] = 1.0
    return idx_ray, locations, normals, idx_tri, mask.reshape(H, W)


def cast_shadows(mesh, hit_locs, hit_normals, light_pos_world, point_light=True):
    """Shadow ray test: returns (N_hits,) mask, 1=lit, 0=shadowed."""
    n_hits = len(hit_locs)
    if n_hits == 0:
        return np.ones(0)

    to_light = light_pos_world[None, :] - hit_locs
    dist_to_light = np.linalg.norm(to_light, axis=1)
    light_dir = to_light / (dist_to_light[:, None] + 1e-8)

    # Offset origins along normal to avoid self-intersection
    origins = hit_locs + hit_normals * 1e-4

    shadow_mask = np.ones(n_hits)
    locations, idx_ray, _ = mesh.ray.intersects_location(
        origins, light_dir, multiple_hits=False
    )
    if len(idx_ray) > 0:
        hit_dists = np.linalg.norm(locations - origins[idx_ray], axis=1)
        if point_light:
            shadowed = hit_dists < dist_to_light[idx_ray] - 1e-3
        else:
            shadowed = np.ones(len(idx_ray), dtype=bool)
        shadow_mask[idx_ray[shadowed]] = 0.0

    return shadow_mask


def shade_directional(hit_rays, hit_normals, light_dir_world, albedo, resolution):
    """Lambertian shading with a directional light.
    albedo: (3,) uniform or (N_hits, 3) per-hit."""
    H, W = resolution, resolution
    image = np.zeros((H * W, 3), dtype=np.float64)
    if len(hit_rays) == 0:
        return image.reshape(H, W, 3)
    shading = np.maximum(0.0, np.sum(hit_normals * light_dir_world, axis=1))
    if albedo.ndim == 1:
        image[hit_rays] = albedo[None, :] * shading[:, None]
    else:
        image[hit_rays] = albedo * shading[:, None]
    return image.reshape(H, W, 3)


def shade_point(hit_rays, hit_locs, hit_normals, light_pos_world, albedo, resolution):
    """Lambertian shading with point light: per-pixel L + 1/r² falloff.
    albedo: (3,) uniform or (N_hits, 3) per-hit."""
    H, W = resolution, resolution
    image = np.zeros((H * W, 3), dtype=np.float64)
    if len(hit_rays) == 0:
        return image.reshape(H, W, 3)
    to_light = light_pos_world[None, :] - hit_locs
    dist_sq = np.sum(to_light ** 2, axis=1, keepdims=True)
    light_dir = to_light / (np.sqrt(dist_sq) + 1e-8)
    attenuation = 1.0 / (dist_sq + 1e-4)
    shading = np.maximum(0.0, np.sum(hit_normals * light_dir, axis=1, keepdims=True))
    if albedo.ndim == 1:
        image[hit_rays] = albedo[None, :] * (shading * attenuation)
    else:
        image[hit_rays] = albedo * (shading * attenuation)
    return image.reshape(H, W, 3)


def generate_rays(K, c2w_cv, resolution):
    """Generate ray origins and directions from OpenCV c2w."""
    H, W = resolution, resolution
    ray_origins = c2w_cv[:3, 3]
    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    K_inv = np.linalg.inv(K)
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    dirs_cam = np.einsum('ij,hwj->hwi', K_inv, pixels)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    R = c2w_cv[:3, :3]
    dirs_world = np.einsum('ij,hwj->hwi', R, dirs_cam)
    return ray_origins, dirs_world


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_camera_setup(positions, mesh, output_path):
    """3D plot of mesh wireframe + camera positions."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2],
               c='gray', s=0.5, alpha=0.3)

    cam_pos = np.array(positions)
    ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
               c='red', s=40, marker='^', label=f'{len(positions)} cameras')
    for i, p in enumerate(positions):
        direction = -p / np.linalg.norm(p) * 0.3
        ax.quiver(p[0], p[1], p[2], direction[0], direction[1], direction[2],
                  color='red', alpha=0.5, arrow_length_ratio=0.2)

    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi, 20)
    r = 1.0
    xs = r * np.outer(np.cos(u_s), np.sin(v_s))
    ys = r * np.outer(np.sin(u_s), np.sin(v_s))
    zs = r * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_wireframe(xs, ys, zs, color='blue', alpha=0.05, linewidth=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Setup ({len(positions)} views)')
    ax.legend()
    max_r = np.max(np.abs(cam_pos)) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_light_rig_directional(light_dirs_gl, output_path):
    """Plot directional light directions on hemisphere.
    light_dirs_gl: (N, 3) in OpenGL camera space."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi / 2, 15)
    xs = np.outer(np.cos(u_s), np.sin(v_s))
    ys = np.outer(np.sin(u_s), np.sin(v_s))
    zs = np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_wireframe(xs, ys, zs, color='lightblue', alpha=0.3, linewidth=0.5)

    # Convert GL to OpenCV for display: flip Y,Z
    dirs_cv = light_dirs_gl.copy()
    dirs_cv[:, [1, 2]] = -dirs_cv[:, [1, 2]]
    dirs_cv = dirs_cv / np.linalg.norm(dirs_cv, axis=1, keepdims=True)

    ax.quiver(np.zeros(len(dirs_cv)), np.zeros(len(dirs_cv)), np.zeros(len(dirs_cv)),
              dirs_cv[:, 0], dirs_cv[:, 1], dirs_cv[:, 2],
              color='orange', arrow_length_ratio=0.1, linewidth=2)
    ax.scatter(dirs_cv[:, 0], dirs_cv[:, 1], dirs_cv[:, 2],
               c='orange', s=50, label=f'{len(dirs_cv)} lights')

    for i, d in enumerate(dirs_cv):
        ax.text(d[0]*1.1, d[1]*1.1, d[2]*1.1, str(i), fontsize=7)

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Y (down)')
    ax.set_zlabel('Z (forward)')
    ax.set_title(f'Directional Light Rig ({len(dirs_cv)} lights, camera space)')
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.2, 1.2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_light_rig_point(led_positions_cv, output_path):
    """Plot point light LED positions near camera.
    led_positions_cv: (N, 3) in OpenCV camera space."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Camera at origin
    ax.scatter([0], [0], [0], c='black', s=100, marker='s', label='Camera')

    ax.scatter(led_positions_cv[:, 0], led_positions_cv[:, 1], led_positions_cv[:, 2],
               c='yellow', edgecolors='orange', s=80, label=f'{len(led_positions_cv)} LEDs')

    for i, p in enumerate(led_positions_cv):
        ax.text(p[0], p[1], p[2] + 0.01, str(i), fontsize=7)

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Y (down)')
    ax.set_zlabel('Z (forward)')
    ax.set_title(f'Point Light LED Rig ({len(led_positions_cv)} LEDs, camera space)')
    ax.legend()
    max_r = np.max(np.abs(led_positions_cv)) * 1.5
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-0.05, max_r * 2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_light_rig_world(light_positions_world, mesh, output_path):
    """Plot fixed world-space point lights on hemisphere with mesh."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2],
               c='gray', s=0.5, alpha=0.3)

    ax.scatter(light_positions_world[:, 0], light_positions_world[:, 1],
               light_positions_world[:, 2],
               c='yellow', edgecolors='orange', s=80,
               label=f'{len(light_positions_world)} point lights')

    for i, p in enumerate(light_positions_world):
        ax.text(p[0], p[1], p[2], str(i), fontsize=7)

    r = np.linalg.norm(light_positions_world, axis=1).mean()
    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi / 2, 15)
    xs = r * np.outer(np.cos(u_s), np.sin(v_s))
    ys = r * np.outer(np.sin(u_s), np.sin(v_s))
    zs = r * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_wireframe(xs, ys, zs, color='lightblue', alpha=0.2, linewidth=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (up)')
    ax.set_title(f'Fixed Point Lights ({len(light_positions_world)} lights, world space, r={r:.1f})')
    ax.legend()
    max_r = r * 1.2
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-0.5, max_r)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_sample_images(output_dir, n_views, n_lights, n_show_views=5, n_show_lights=4):
    """Grid of sample rendered images."""
    view_indices = np.round(np.linspace(0, n_views - 1, n_show_views)).astype(int)
    light_indices = np.round(np.linspace(0, n_lights - 1, n_show_lights)).astype(int)

    fig, axes = plt.subplots(n_show_views, n_show_lights,
                              figsize=(4 * n_show_lights, 4 * n_show_views))
    for i, vi in enumerate(view_indices):
        for j, li in enumerate(light_indices):
            fname = os.path.join(output_dir, "image", f"V{vi:02d}L{li:02d}.png")
            img = cv.imread(fname)
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
            axes[i, j].set_title(f"V{vi:02d}L{li:02d}", fontsize=9)
            axes[i, j].axis('off')

    plt.suptitle(f'Sample Images ({n_views} views x {n_lights} lights)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def compute_image_scores(image_rgb, mask_bool):
    """Compute per-image selection scores (mean gray, mean Sobel magnitude) within mask.

    Args:
        image_rgb: (H, W, 3) float64 in [0, 1]
        mask_bool: (H, W) bool

    Returns:
        mean_gray: mean luminance in masked region
        mean_sobel: mean Sobel gradient magnitude in masked region
    """
    gray = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    n_pixels = mask_bool.sum()
    if n_pixels == 0:
        return 0.0, 0.0
    mean_gray = float(gray[mask_bool].mean())
    # Sobel magnitude
    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    sx = cv.Sobel(gray_u8, cv.CV_64F, 1, 0, ksize=3)
    sy = cv.Sobel(gray_u8, cv.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx ** 2 + sy ** 2)
    mean_sobel = float(sobel_mag[mask_bool].mean())
    return mean_gray, mean_sobel


def generate_strategy_indices(output_dir, n_views, n_lights,
                               gray_matrix, sobel_matrix,
                               train_views, val_views, test_views):
    """Generate per-strategy view_light_idx files and sample image plots.

    Strategies (1 light selected per view, or fixed for all views):
        maxgray   : light with highest mean gray per view  (well-lit, shadows visible)
        mingray   : light with lowest mean gray per view   (grazing, dark)
        medgray   : light closest to median gray per view  (balanced)
        maxsobel  : light with highest Sobel contrast per view (most informative for geometry)
        1l_front  : always L00 (frontal LED, low relief)
        1l_maxvar : single fixed Lk maximizing std of per-view mean gray (best photometric diversity)
    """
    # ── Per-view light selection ───────────────────────────────────────────────
    sel_maxgray = np.argmax(gray_matrix, axis=1)
    sel_mingray = np.argmin(gray_matrix, axis=1)
    sel_medgray = np.array([
        np.argmin(np.abs(gray_matrix[vi] - np.median(gray_matrix[vi])))
        for vi in range(n_views)
    ])
    sel_maxsobel = np.argmax(sobel_matrix, axis=1)

    # ── Fixed single-light strategies ─────────────────────────────────────────
    sel_1l_front = np.zeros(n_views, dtype=int)   # always L00
    gray_std_per_light = gray_matrix.std(axis=0)  # (n_lights,)
    best_light = int(np.argmax(gray_std_per_light))
    sel_1l_maxvar = np.full(n_views, best_light, dtype=int)
    print(f"  Strategy 1l_maxvar: best light = L{best_light:02d} "
          f"(cross-view gray std={gray_std_per_light[best_light]:.4f})")

    strategies = {
        'maxgray':   sel_maxgray,
        'mingray':   sel_mingray,
        'medgray':   sel_medgray,
        'maxsobel':  sel_maxsobel,
        '1l_front':  sel_1l_front,
        '1l_maxvar': sel_1l_maxvar,
    }

    splits = [("train", train_views), ("val", val_views), ("test", test_views)]

    for strat_name, light_per_view in strategies.items():
        for split_name, view_list in splits:
            indices = [f"V{vi:02d}L{int(light_per_view[vi]):02d}" for vi in view_list]
            fpath = os.path.join(output_dir, f"view_light_idx_{strat_name}_{split_name}.txt")
            with open(fpath, 'w') as f:
                f.write('\n'.join(indices) + '\n')

        uniq, cnts = np.unique(light_per_view, return_counts=True)
        top = sorted(zip(cnts, uniq), reverse=True)[:3]
        top_str = ', '.join([f"L{l:02d}×{c}" for c, l in top])
        print(f"  Strategy '{strat_name}': 3 split files saved. Top lights: {top_str}")

        plot_strategy_sample_images(output_dir, strat_name, light_per_view,
                                    gray_matrix, sobel_matrix, n_views, n_lights)


def plot_strategy_sample_images(output_dir, strat_name, light_per_view,
                                 gray_matrix, sobel_matrix,
                                 n_views, n_lights, n_show=8):
    """Grid showing the selected (view, light) image per view, with score annotations
    and a per-view inset bar chart of gray scores across lights.
    """
    show_views = np.round(np.linspace(0, n_views - 1, n_show)).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4.5))
    if n_show == 1:
        axes = [axes]

    for col, vi in enumerate(show_views):
        li = int(light_per_view[vi])
        fname = os.path.join(output_dir, "image", f"V{vi:02d}L{li:02d}.png")
        ax = axes[col]
        img = cv.imread(fname)
        if img is not None:
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        else:
            ax.set_facecolor('black')

        mg = gray_matrix[vi, li]
        ms = sobel_matrix[vi, li]
        ax.set_title(f"V{vi:02d}·L{li:02d}\n☼{mg:.3f} ∇{ms:.1f}", fontsize=8)
        ax.axis('off')

        # Inset bar chart: gray per light for this view, selected one in red
        ax_ins = ax.inset_axes([0.0, -0.28, 1.0, 0.22])
        bar_colors = ['#888888'] * n_lights
        bar_colors[li] = '#ff3333'
        ax_ins.bar(range(n_lights), gray_matrix[vi], color=bar_colors, width=1.0)
        ax_ins.set_xlim(-0.5, n_lights - 0.5)
        ax_ins.set_yticks([])
        ax_ins.set_xticks([0, n_lights - 1])
        ax_ins.tick_params(labelsize=6)
        ax_ins.set_xlabel('L idx', fontsize=6)
        ax_ins.patch.set_alpha(0.0)

    fig.suptitle(f"Strategy '{strat_name}' — selected (view, light) pairs\n"
                 f"(☼=mean gray  ∇=mean Sobel)", fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(output_dir, f"sample_images_{strat_name}.png")
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_dataset_overview(output_dir, positions, mesh, light_type, n_lights):
    """Combined overview: setup + sample images."""
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121, projection='3d')
    vertices = mesh.vertices
    ax1.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2],
                c='gray', s=0.5, alpha=0.3)
    cam_pos = np.array(positions)
    ax1.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
                c='red', s=30, marker='^')
    ax1.set_title(f'Camera Setup ({len(positions)} views)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122)
    sample_imgs = []
    for vi in [0, len(positions) // 4, len(positions) // 2]:
        for li in [0, n_lights // 2]:
            fname = os.path.join(output_dir, "image", f"V{vi:02d}L{li:02d}.png")
            img = cv.imread(fname)
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_small = cv.resize(img, (200, 200))
                sample_imgs.append(img_small)
    if sample_imgs:
        grid = np.concatenate([
            np.concatenate(sample_imgs[:3], axis=1),
            np.concatenate(sample_imgs[3:6] if len(sample_imgs) >= 6 else sample_imgs[:3], axis=1)
        ], axis=0)
        ax2.imshow(grid)
    ax2.set_title(f'{light_type} light, {n_lights} lights')
    ax2.axis('off')

    plt.suptitle(f'Dataset Overview: synthetic_suzanne_{light_type}', fontsize=16)
    out_path = os.path.join(output_dir, "dataset_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Suzanne dataset for MVSCPS")
    parser.add_argument("--mesh_path", type=str,
                        default=os.path.expanduser("~/dev/RNb-NeuS/data/synthetic_suzanne/gt_mesh.ply"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_views", type=int, default=50)
    parser.add_argument("--n_lights", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--light_type", type=str, default="directional", choices=["directional", "point"])
    parser.add_argument("--light_distance", type=float, default=0.15,
                        help="LED-to-camera distance for camera-mounted point lights")
    parser.add_argument("--light_radius", type=float, default=2.0,
                        help="Radius of fixed point light hemisphere in world space")
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--n_rings", type=int, default=3)
    parser.add_argument("--camera_sampling", type=str, default="rings",
                        choices=["rings", "fibonacci"],
                        help="Camera distribution: rings (legacy) or fibonacci (full sphere)")
    parser.add_argument("--albedo", type=float, nargs=3, default=[0.7, 0.7, 0.7])
    parser.add_argument("--colored_albedo", action="store_true",
                        help="Patchwork colored albedo (Elmer-style)")
    parser.add_argument("--n_patches", type=int, default=12)
    parser.add_argument("--shadows", action="store_true",
                        help="Enable ray-traced cast shadows")
    parser.add_argument("--train_views", type=int, default=40)
    parser.add_argument("--val_views", type=int, default=5)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    if args.shadows:
        os.makedirs(os.path.join(output_dir, "shadow"), exist_ok=True)

    # --- Load mesh ---
    print(f"Loading mesh from: {args.mesh_path}")
    mesh = trimesh.load(args.mesh_path, process=False)
    print(f"  Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")
    mesh_max_radius = np.linalg.norm(mesh.vertices, axis=1).max()
    print(f"  Max radius: {mesh_max_radius:.4f}")

    gt_dest = os.path.join(output_dir, "gt_mesh.ply")
    if os.path.abspath(args.mesh_path) != os.path.abspath(gt_dest):
        shutil.copy2(args.mesh_path, gt_dest)

    # --- O2W: bounding sphere with margin ---
    mesh_center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2.0
    mesh_radius = np.linalg.norm(mesh.vertices - mesh_center, axis=1).max()
    O2W_scale = float(mesh_radius / 0.7)
    O2W_translation = mesh_center.tolist()
    print(f"  O2W_scale: {O2W_scale:.4f}, O2W_translation: {O2W_translation}")
    print(f"  Mesh max radius in obj_space: {mesh_radius / O2W_scale:.4f}")

    # --- Intrinsics ---
    res = args.resolution
    focal_mm = 50.0
    sensor_w = 36.0
    fx = focal_mm * res / sensor_w
    fy = fx
    cx = res / 2.0
    cy = res / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # --- Camera positions ---
    if args.camera_sampling == "fibonacci":
        positions = generate_camera_positions_fibonacci(args.n_views, args.camera_distance)
        print(f"\n{len(positions)} camera positions generated (Fibonacci full sphere)")
    else:
        positions = generate_camera_positions(args.n_views, args.n_rings, args.camera_distance)
        print(f"\n{len(positions)} camera positions generated (rings)")

    # --- Albedo ---
    if args.colored_albedo:
        face_colors = generate_patchwork_albedo(mesh, n_patches=args.n_patches)
        print(f"Colored albedo: {args.n_patches} patches")
        albedo_uniform = None
    else:
        face_colors = None
        albedo_uniform = np.array(args.albedo, dtype=np.float64)

    # --- Light rig ---

    if args.light_type == "directional":
        light_dirs_gl = generate_light_directions_hemisphere(args.n_lights)
        print(f"{args.n_lights} directional light directions generated (OpenGL cam space)")
    else:
        led_positions_cv = generate_led_positions(args.n_lights, args.light_distance)
        light_intensity_value = args.light_distance ** 2
        print(f"{args.n_lights} camera-mounted LEDs (OpenCV cam space, dist={args.light_distance})")
        print(f"  Light intensity = {light_intensity_value:.1f} (compensates 1/r² at nominal dist)")

    # --- Render loop ---
    cam_dict = {
        "O2W_scale": O2W_scale,
        "O2W_translation": O2W_translation,
    }
    print(f"\n=== Rendering {len(positions)} views x {args.n_lights} lights ===\n")

    # --- Score matrices for view-light selection strategies ---
    gray_matrix = np.zeros((len(positions), args.n_lights), dtype=np.float64)
    sobel_matrix = np.zeros((len(positions), args.n_lights), dtype=np.float64)

    all_masks = {}
    for vi, cam_pos in enumerate(positions):
        c2w_blender = look_at(cam_pos)
        c2w_cv = blender_c2w_to_opencv_c2w(c2w_blender)
        R_c2w = c2w_cv[:3, :3]
        cam_center = c2w_cv[:3, 3]

        ray_origins, dirs_world = generate_rays(K, c2w_cv, res)

        # Single ray-cast per view (expensive part, done once)
        hit_rays, hit_locs, hit_normals, hit_tri, mask = raycast_view(mesh, ray_origins, dirs_world, res)
        all_masks[vi] = mask > 0.5

        # Per-hit albedo
        if face_colors is not None:
            albedo = face_colors[hit_tri] if len(hit_tri) > 0 else np.zeros((0, 3))
        else:
            albedo = albedo_uniform

        # Fast shading per light (cheap, vectorized numpy)
        for li in range(args.n_lights):
            vl_idx = f"V{vi:02d}L{li:02d}"
            cam_dict[f"K_{vl_idx}"] = K.tolist()
            cam_dict[f"C2W_{vl_idx}"] = c2w_cv[:3].tolist()

            if args.light_type == "directional":
                light_dir_cv = light_dirs_gl[li].copy()
                light_dir_cv[[1, 2]] = -light_dir_cv[[1, 2]]
                light_dir_world = R_c2w @ light_dir_cv
                light_dir_world = light_dir_world / np.linalg.norm(light_dir_world)
                image = shade_directional(hit_rays, hit_normals, light_dir_world, albedo, res)
                if args.shadows and len(hit_rays) > 0:
                    far_light = hit_locs.mean(axis=0) + light_dir_world * 100.0
                    shadow = cast_shadows(mesh, hit_locs, hit_normals, far_light, point_light=False)
                    image_flat = image.reshape(-1, 3)
                    image_flat[hit_rays] *= shadow[:, None]
                    image = image_flat.reshape(res, res, 3)
            else:
                light_pos_world = R_c2w @ led_positions_cv[li] + cam_center
                image = shade_point(hit_rays, hit_locs, hit_normals,
                                    light_pos_world, albedo, res)
                image *= light_intensity_value
                if args.shadows and len(hit_rays) > 0:
                    shadow = cast_shadows(mesh, hit_locs, hit_normals, light_pos_world, point_light=True)
                    image_flat = image.reshape(-1, 3)
                    image_flat[hit_rays] *= shadow[:, None]
                    image = image_flat.reshape(res, res, 3)

            # Save shadow mask
            if args.shadows and len(hit_rays) > 0:
                shadow_img = np.zeros((res * res,), dtype=np.uint8)
                shadow_img[hit_rays] = (shadow * 255).astype(np.uint8)
                cv.imwrite(
                    os.path.join(output_dir, "shadow", f"{vl_idx}.png"),
                    shadow_img.reshape(res, res),
                    [cv.IMWRITE_PNG_COMPRESSION, 3]
                )

            img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

            # Collect selection scores (inline, no extra I/O)
            mg, ms = compute_image_scores(image, mask > 0.5)
            gray_matrix[vi, li] = mg
            sobel_matrix[vi, li] = ms

            cv.imwrite(
                os.path.join(output_dir, "image", f"{vl_idx}.png"),
                cv.cvtColor(img_uint8, cv.COLOR_RGB2BGR),
                [cv.IMWRITE_PNG_COMPRESSION, 3]
            )

        print(f"  View {vi:02d}/{len(positions)-1} rendered ({args.n_lights} lights)")

    # --- Save masks ---
    for vi, mask_bool in all_masks.items():
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        cv.imwrite(
            os.path.join(output_dir, "mask", f"V{vi:02d}.png"),
            mask_uint8, [cv.IMWRITE_PNG_COMPRESSION, 3]
        )

    # --- Save camera_params.json ---
    cam_json_path = os.path.join(output_dir, "camera_params.json")
    with open(cam_json_path, 'w') as f:
        json.dump(cam_dict, f, indent=2)
    print(f"\nSaved {cam_json_path}")

    # --- Save light files ---
    if args.light_type == "directional":
        intensities = np.ones((args.n_lights, 3))
        np.savetxt(os.path.join(output_dir, "light_directions.txt"),
                   light_dirs_gl, fmt="%.8f")
        print(f"Saved light_directions.txt ({args.n_lights} directions, OpenGL cam space)")
    else:
        intensities = np.ones((args.n_lights, 3)) * light_intensity_value
        # Camera-space positions scaled by O2W_scale (MVSCPS convention)
        led_pos_scaled = led_positions_cv / O2W_scale
        np.savez(os.path.join(output_dir, "light_positions_cam_scaled.npz"),
                 light_positions=led_pos_scaled)
        print(f"Saved light_positions_cam_scaled.npz ({args.n_lights} positions, cam-space/O2W_scale)")
        print(f"  LED dist={args.light_distance:.2f} → scaled dist={np.linalg.norm(led_pos_scaled, axis=1).mean():.2f}")

    np.savetxt(os.path.join(output_dir, "light_intensities.txt"), intensities, fmt="%.6f")
    print(f"Saved light_intensities.txt (intensity={intensities[0, 0]:.1f})")

    # --- Save view-light index files ---
    all_views = list(range(len(positions)))
    np.random.seed(42)
    np.random.shuffle(all_views)
    train_views = sorted(all_views[:args.train_views])
    val_views = sorted(all_views[args.train_views:args.train_views + args.val_views])
    test_views = sorted(all_views[args.train_views + args.val_views:])

    for split_name, view_list in [("train", train_views), ("val", val_views), ("test", test_views)]:
        indices = []
        for vi in view_list:
            for li in range(args.n_lights):
                indices.append(f"V{vi:02d}L{li:02d}")
        fpath = os.path.join(output_dir, f"view_light_idx_{split_name}.txt")
        with open(fpath, 'w') as f:
            f.write('\n'.join(indices) + '\n')
        print(f"Saved {fpath} ({len(indices)} entries, {len(view_list)} views)")

    # --- Save per-strategy view-light index files ---
    print("\n=== Generating view-light selection strategies ===\n")
    generate_strategy_indices(output_dir, len(positions), args.n_lights,
                              gray_matrix, sobel_matrix,
                              train_views, val_views, test_views)

    # --- Plots ---
    if not args.no_plot:
        print("\n=== Generating plots ===\n")
        plot_camera_setup(positions, mesh, os.path.join(output_dir, "camera_setup.png"))
        if args.light_type == "directional":
            plot_light_rig_directional(light_dirs_gl, os.path.join(output_dir, "light_rig.png"))
        else:
            plot_light_rig_point(led_positions_cv, os.path.join(output_dir, "light_rig.png"))
        plot_sample_images(output_dir, len(positions), args.n_lights)
        plot_dataset_overview(output_dir, positions, mesh, args.light_type, args.n_lights)

    print(f"\n=== Dataset generated in {output_dir} ===")
    print(f"  Views:      {len(positions)}")
    print(f"  Lights:     {args.n_lights} ({args.light_type})")
    print(f"  Resolution: {res}x{res}")
    print(f"  Images:     {len(positions) * args.n_lights}")
    print(f"  Albedo:     {albedo}")
    print(f"  O2W_scale:  {O2W_scale:.4f}")
    print(f"  Train/Val/Test views: {len(train_views)}/{len(val_views)}/{len(test_views)}")


if __name__ == "__main__":
    main()
