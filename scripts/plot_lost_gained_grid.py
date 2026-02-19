#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 2x2 grid of Lost & Gained Van Krevelen plots for consistent styling.

Version History:
- v1: Used load_and_prepare_data and compute_lost_gained_shared functions
- v2: Uses load_icrms_xlsx, categorize_formulas, and vk_coords_df_from_formulas directly

Current version: v2 approach (more direct implementation)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import pykrev as pk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from lost_gained_vankrevelen import (
    load_icrms_xlsx,
    to_ms_tuple,
    vk_coords_df_from_formulas,
    categorize_formulas,
    global_axis_limits
)


def main():
    # Clear all matplotlib state to avoid any residual elements
    plt.close('all')

    raw_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'comparisons_grid'
    output_dir.mkdir(exist_ok=True)

    # Load all data
    data_files = sorted(raw_dir.glob('ICRMS-*.xlsx'))
    df_map = {}
    for f in data_files:
        name = f.stem
        df_map[name] = load_icrms_xlsx(f)

    # Define 4 pairs to plot
    pairs = [
        ('ICRMS-1', 'ICRMS-2'),
        ('ICRMS-1', 'ICRMS-3'),
        ('ICRMS-1', 'ICRMS-4'),
        ('ICRMS-2', 'ICRMS-3'),
    ]

    # Compute unified axis limits
    xlim, ylim = global_axis_limits(list(df_map.values()))

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    axes = axes.flatten()

    # Color scheme options (uncomment to use different schemes):
    # Scheme 1: Red-Green-Gray (current)
    color_lost = '#d62728'   # red
    color_gained = '#2ca02c'  # green
    color_shared = '#9e9e9e'  # gray

    # Scheme 2: Red-Blue-LightGray (alternative)
    # color_lost = '#d62728'   # red
    # color_gained = '#1f77b4'  # blue
    # color_shared = 'lightgray'

    # Scheme 3: Orange-Purple-Gray (alternative)
    # color_lost = '#ff7f0e'   # orange
    # color_gained = '#9467bd'  # purple
    # color_shared = '#7f7f7f'  # gray

    # Plot each pair
    for idx, (name_a, name_b) in enumerate(pairs):
        ax = axes[idx]
        df_a = df_map[name_a]
        df_b = df_map[name_b]

        # Categorize formulas
        lost, gained, shared = categorize_formulas(df_a, df_b)

        # Compute VK coordinates for each category
        lost_df = vk_coords_df_from_formulas(lost) if lost else pd.DataFrame()
        gained_df = vk_coords_df_from_formulas(gained) if gained else pd.DataFrame()
        shared_df = vk_coords_df_from_formulas(shared) if shared else pd.DataFrame()

        # Set axis spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Plot shared (background layer)
        # Styling options (uncomment to try different styles):
        # Style 1: Small, very transparent (current)
        if not shared_df.empty:
            ax.scatter(
                shared_df['oc'], shared_df['hc'],
                s=15, c=color_shared, alpha=0.35,
                label=f'Shared (n={len(shared_df)})', edgecolors='none'
            )
        # Style 2: Medium, slightly less transparent
        # if not shared_df.empty:
        #     ax.scatter(
        #         shared_df['oc'], shared_df['hc'],
        #         s=20, c=color_shared, alpha=0.5,
        #         label=f'Shared (n={len(shared_df)})', edgecolors='none'
        #     )
        # Style 3: Larger, more visible
        # if not shared_df.empty:
        #     ax.scatter(
        #         shared_df['oc'], shared_df['hc'],
        #         s=25, c=color_shared, alpha=0.6,
        #         label=f'Shared (n={len(shared_df)})', edgecolors='none'
        #     )

        # Plot lost (foreground layer)
        # Styling options (uncomment to try different styles):
        # Style 1: Large, mostly opaque (current)
        if not lost_df.empty:
            ax.scatter(
                lost_df['oc'], lost_df['hc'],
                s=25, c=color_lost, alpha=0.85,
                label=f'Lost (n={len(lost_df)})', edgecolors='none'
            )
        # Style 2: Very large, fully opaque
        # if not lost_df.empty:
        #     ax.scatter(
        #         lost_df['oc'], lost_df['hc'],
        #         s=35, c=color_lost, alpha=1.0,
        #         label=f'Lost (n={len(lost_df)})', edgecolors='none'
        #     )
        # Style 3: Medium, with edge
        # if not lost_df.empty:
        #     ax.scatter(
        #         lost_df['oc'], lost_df['hc'],
        #         s=20, c=color_lost, alpha=0.9,
        #         label=f'Lost (n={len(lost_df)})', edgecolors='black', linewidth=0.5
        #     )

        # Plot gained (foreground layer)
        # Styling options (uncomment to try different styles):
        # Style 1: Large, mostly opaque (current)
        if not gained_df.empty:
            ax.scatter(
                gained_df['oc'], gained_df['hc'],
                s=25, c=color_gained, alpha=0.85,
                label=f'Gained (n={len(gained_df)})', edgecolors='none'
            )
        # Style 2: Very large, fully opaque
        # if not gained_df.empty:
        #     ax.scatter(
        #         gained_df['oc'], gained_df['hc'],
        #         s=35, c=color_gained, alpha=1.0,
        #         label=f'Gained (n={len(gained_df)})', edgecolors='none'
        #     )
        # Style 3: Medium, with edge
        # if not gained_df.empty:
        #     ax.scatter(
        #         gained_df['oc'], gained_df['hc'],
        #         s=20, c=color_gained, alpha=0.9,
        #         label=f'Gained (n={len(gained_df)})', edgecolors='black', linewidth=0.5
        #     )

        # Styling
        ax.set_xlabel('O/C', fontsize=18, fontweight='bold')
        ax.set_ylabel('H/C', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{name_a} vs {name_b}', fontsize=16, fontweight='bold')
        ax.grid(False)

        # Legend styling options (uncomment to try different styles):
        # Style 1: No frame, standard size (current)
        legend = ax.legend(loc='best', frameon=False, fontsize=14)
        # Style 2: With frame, slightly smaller
        # legend = ax.legend(loc='best', frameon=True, fontsize=12)
        # Style 3: Outside the plot
        # legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=14)

        for text in legend.get_texts():
            text.set_fontweight('bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'lost_gained_grid.png', dpi=200)
    plt.close(fig)
    print(f'[INFO] Saved: {output_dir / "lost_gained_grid.png"}')


if __name__ == '__main__':
    main()
