# code from "https://github.com/guanyingc/UPS-GCNet/blob/master/utils/draw_utils.py"
import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib; matplotlib.use('agg')

def set_figscale(fig, ax):
    x0, y0, dx, dy = ax.get_position().bounds
    w = 3 * max(dx, dy) /dx
    h = 3 * max(dx, dy) /dy
    fig.set_size_inches((w, h))

def draw_circle(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    x, y = np.cos(t), np.sin(t)
    ax.plot(x*1.0, y*1.0, 'k')
    axis = 1.01
    ax.axis([-axis, axis, -axis, axis])

def plot_light(x, y, save_name, c=None):
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(x, y, s=6)
        # add text annotation
        # for i in range(x.shape[0]):
        #     ax.text(x[i], y[i], str(i), fontsize=6)
    else:
        plt.scatter(x, y, c=c, cmap='jet', vmin=0, vmax=1)
        # add text annotation
        # for i in range(x.shape[0]):
        #     ax.text(x[i], y[i], str(i), fontsize=6)

    draw_circle(ax)

    ax.axis('off')
    set_figscale(fig, ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(save_name, bbox_inches=extent, transparent=True)
    plt.close()

def plot_lighting(dirs, ints, save_dir):
    # Visualize light direction and intensity
    save_name = os.path.join(save_dir, 'est_light_map.png')
    if len(ints.shape) > 1:
        ints = ints.mean(-1)
    ints = ints / ints.max()
    plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

def plot_lighting_gt(dirs, ints, save_dir):
    # Visualize light direction and intensity
    save_name = os.path.join(save_dir, 'light_map_gt.pdf')
    if len(ints.shape) > 1:
        ints = ints.mean(-1)
    ints = ints / ints.max()
    plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

def plot_light_pos_3d(positions, save_name, c=None, gt_positions=None):
    """3D scatter of point light positions in camera space."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], c='black', s=100, marker='s', label='Camera')
    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=c if c is not None else 'orange', cmap='jet',
                    s=60, edgecolors='k', linewidth=0.5, label='Estimated')
    for i, p in enumerate(positions):
        ax.text(p[0], p[1], p[2], str(i), fontsize=6)
    if gt_positions is not None:
        ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                   c='green', s=40, marker='^', alpha=0.5, label='GT')
    r = max(np.abs(positions).max(), 0.5) * 1.3
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-0.1, r)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (forward)')
    ax.set_title(f'Point lights ({len(positions)})')
    ax.legend(fontsize=8)
    plt.savefig(save_name, dpi=100, bbox_inches='tight')
    plt.close()


def plot_dir_error(light, error, save_dir):
    # plot light direction estimation error
    save_name = os.path.join(save_dir, 'est_light_error_dir.png')
    error = error / 25
    plot_light(light[:,0], light[:, 1], save_name, error)

def plot_int_error(light, error, save_dir):
    # plot light intensity estimation error
    save_name = os.path.join(save_dir, 'est_light_error_int.png')
    error = error / 0.2
    plot_light(light[:,0], light[:, 1], save_name, error)