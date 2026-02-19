#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 2x2 grid of Density Van Krevelen plots for consistent styling.

Version History (see comments below for alternative implementations):
- v1: Basic pykrev van_krevelen_plot with density coloring
- v2: Added unified axis limits via compute_global_limits
- v3: Same as v2 (minimal change)
- v4: Same as v2 (minimal change)
- v5: Uses gaussian_kde directly with element_ratios for coordinate computation

Current version: v5 approach (direct gaussian_kde with element_ratios)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import pykrev as pk
from scipy.stats import gaussian_kde

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from batch_export_ftms import load_icrms_xlsx
from lost_gained_vankrevelen import vk_coords_df_from_formulas


def compute_global_limits_from_formulas(data_files):
    """Compute unified axis limits from all data files using element_ratios."""
    all_oc = []
    all_hc = []

    for f in data_files:
        df = load_icrms_xlsx(f)
        formulas = df['formula'].dropna().astype(str).tolist()
        coords_df = vk_coords_df_from_formulas(formulas)

        if not coords_df.empty:
            all_oc.extend(coords_df['oc'].values)
            all_hc.extend(coords_df['hc'].values)

    if not all_oc or not all_hc:
        return (0.0, 1.2), (0.0, 2.2)

    oc = np.array(all_oc)
    hc = np.array(all_hc)
    oc = oc[np.isfinite(oc)]
    hc = hc[np.isfinite(hc)]

    # Compute limits with padding
    xr = (np.nanmax(oc) - np.nanmin(oc)) if np.nanmax(oc) - np.nanmin(oc) > 0 else 0.1
    yr = (np.nanmax(hc) - np.nanmin(hc)) if np.nanmax(hc) - np.nanmin(hc) > 0 else 0.1

    xlim = (max(0.0, np.nanmin(oc) - 0.05 * xr), np.nanmax(oc) + 0.05 * xr)
    ylim = (max(0.0, np.nanmin(hc) - 0.05 * yr), np.nanmax(hc) + 0.05 * yr)

    return xlim, ylim


def compute_density_limits(data_files):
    """Compute unified density limits from all data files."""
    all_density = []

    for f in data_files:
        df = load_icrms_xlsx(f)
        formulas = df['formula'].dropna().astype(str).tolist()
        coords_df = vk_coords_df_from_formulas(formulas)

        if not coords_df.empty:
            oc = coords_df['oc'].values
            hc = coords_df['hc'].values
            try:
                xy = np.vstack([oc, hc])
                z = gaussian_kde(xy)(xy)
                all_density.extend(z.tolist())
            except Exception:
                pass

    if not all_density:
        return (0.0, 1.0)

    density = np.array(all_density)
    density = density[np.isfinite(density)]

    dr = (np.nanmax(density) - np.nanmin(density)) if np.nanmax(density) - np.nanmin(density) > 0 else 0.1
    density_limits = (np.nanmin(density), np.nanmax(density))

    return density_limits


"""
# ============================================================================
# ALTERNATIVE IMPLEMENTATIONS (from previous versions)
# ============================================================================

# ------------------- v1-v4: Using pykrev's van_krevelen_plot -------------------
# This approach uses pykrev's built-in density computation:
#
# def compute_global_limits(data_files):
#     """Pre-scan all datasets to get unified limits for plotting."""
#     y_min, y_max = np.inf, -np.inf
#     dbe_min, dbe_max = np.inf, -np.inf
#     dens_min, dens_max = np.inf, -np.inf
#
#     for xlsx_path in data_files:
#         if not Path(xlsx_path).exists():
#             continue
#         df = load_icrms_xlsx(Path(xlsx_path))
#         msTuple = to_ms_tuple(df)
#
#         # Density limits via temporary VK plot
#         try:
#             fig, ax = pk.van_krevelen_plot(msTuple, y_ratio='HC', c='density', s=7)
#             scatter = ax.collections[0] if ax.collections else None
#             if scatter is not None:
#                 # y from offsets
#                 offsets = scatter.get_offsets()
#                 pts = np.asarray(offsets)
#                 if pts.size:
#                     ys = pts[:, 1]
#                     ys = ys[np.isfinite(ys)]
#                     if ys.size:
#                         y_min = min(y_min, float(np.min(ys)))
#                         y_max = max(y_max, float(np.max(ys)))
#                 # density range from color array
#                 arr = np.asarray(scatter.get_array())
#                 arr = arr[np.isfinite(arr)]
#                 if arr.size:
#                     dens_min = min(dens_min, float(np.min(arr)))
#                     dens_max = max(dens_max, float(np.max(arr)))
#             plt.close(fig)
#         except Exception:
#             pass
#
#     if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
#         y_lims = (0.0, 2.0)
#     else:
#         y_lims = (y_min, y_max)
#
#     if not np.isfinite(dens_min) or not np.isfinite(dens_max) or dens_min >= dens_max:
#         density_limits = (0.0, 1.0)
#     else:
#         density_limits = (dens_min, dens_max)
#
#     return y_lims, density_limits
#
# And in the plotting section:
#
# # Create plot using pykrev on a temporary figure
# temp_fig, temp_ax = pk.van_krevelen_plot(ms_obj, y_ratio='HC', c='density', s=7)
# scatter = temp_ax.collections[0] if temp_ax.collections else None
#
# # Get data from the scatter plot
# if scatter is not None:
#     offsets = scatter.get_offsets()
#     oc = offsets[:, 0]
#     hc = offsets[:, 1]
#     z = scatter.get_array()
#
#     # Replot on our axes
#     new_scatter = ax.scatter(oc, hc, c=z, s=7, cmap='viridis', alpha=0.7, edgecolors='none')
#     new_scatter.set_clim(density_limits[0], density_limits[1])
#     plt.close(temp_fig)

# ------------------- STYLING OPTIONS -------------------
# Different visual styles tried across versions:

# Style 1 (v1): Light, transparent
# scatter = ax.scatter(oc, hc, c=z, s=7, cmap='viridis', alpha=0.7, edgecolors='none')

# Style 2 (v5): Dense, opaque, larger points
# scatter = ax.scatter(oc_sorted, hc_sorted, c=z_sorted, s=15, cmap='viridis', alpha=1.0)

# Style 3: Medium transparency, medium size
# scatter = ax.scatter(oc, hc, c=z, s=10, cmap='viridis', alpha=0.85, edgecolors='none')

# Color map alternatives:
# cmap='viridis'  # Default (current)
# cmap='plasma'   # Purple-orange
# cmap='inferno'  # Black-red-yellow
# cmap='magma'    # Black-purple-pink
# cmap='cividis'  # Colorblind-friendly
"""


def main():
    # Clear all matplotlib state
    plt.close('all')

    raw_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'density_grid'
    output_dir.mkdir(exist_ok=True)

    # Load all data
    data_files = sorted(raw_dir.glob('ICRMS-*.xlsx'))

    # Compute unified limits using the same method as lost_gained grid
    xlim, ylim = compute_global_limits_from_formulas(data_files)
    density_limits = compute_density_limits(data_files)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    axes = axes.flatten()

    # Plot each dataset
    for idx, f in enumerate(data_files):
        ax = axes[idx]
        name = f.stem.replace('ICRMS-', '')

        # Load data and compute coordinates using element_ratios
        df = load_icrms_xlsx(f)
        formulas = df['formula'].dropna().astype(str).tolist()
        coords_df = vk_coords_df_from_formulas(formulas)

        if not coords_df.empty:
            oc = coords_df['oc'].values
            hc = coords_df['hc'].values

            # Compute density
            try:
                xy = np.vstack([oc, hc])
                z = gaussian_kde(xy)(xy)

                # Sort by density so high-density points are plotted on top (prevents occlusion by low-density points)
                idx_sorted = z.argsort()
                oc_sorted, hc_sorted, z_sorted = oc[idx_sorted], hc[idx_sorted], z[idx_sorted]

                # Plot with v5 styling (dense, opaque, larger points)
                # Alternative styling options (uncomment to try different styles):
                # scatter = ax.scatter(oc_sorted, hc_sorted, c=z_sorted, s=7, cmap='viridis', alpha=0.7, edgecolors='none')
                # scatter = ax.scatter(oc_sorted, hc_sorted, c=z_sorted, s=10, cmap='plasma', alpha=0.85, edgecolors='none')
                # scatter = ax.scatter(oc_sorted, hc_sorted, c=z_sorted, s=20, cmap='inferno', alpha=1.0)
                scatter = ax.scatter(oc_sorted, hc_sorted, c=z_sorted, s=15, cmap='viridis', alpha=1.0)
                scatter.set_clim(density_limits[0], density_limits[1])

                # Colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Kernel Density', fontsize=14, fontweight='bold')
                for label in cbar.ax.get_yticklabels():
                    label.set_fontweight('bold')
            except Exception as e:
                print(f'[WARN] Density computation failed for {name}: {e}')

        # Set axis limits (same as lost_gained grid)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.set_xlabel('O/C', fontsize=18, fontweight='bold')
        ax.set_ylabel('H/C', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.set_title(f'ICRMS-{name}', fontsize=16, fontweight='bold')
        ax.grid(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'density_grid.png', dpi=200)
    plt.close(fig)
    print(f'[INFO] Saved: {output_dir / "density_grid.png"}')


if __name__ == '__main__':
    main()
