#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lost & Gained Van Krevelen plots for ICRMS datasets.

Environment: Batch Processing Environment (Python 3.11)
 - Assumes dependencies: numpy, pandas, matplotlib, openpyxl, pykrev
 - Uses pykrev to derive Van Krevelen coordinates (O/C, H/C) for consistency across scripts

Functionality
 - Load Excel files from data/raw/*.xlsx with at least column: Formula (m/z, Intensity optional)
- For each requested pair (A,B), compute three sets by molecular formula:
  * Lost   = present in A only (A \ B)
  * Gained = present in B only (B \ A)
  * Shared = present in both (A ∩ B)
 - Compute O/C and H/C via pykrev by building msTuple objects (Spectrum optional)
- Plot a single Van Krevelen (x: O/C, y: H/C) with three categories in different colors
- Unify axis limits across all available input files
- Save figure (PNG) and optional tables (XLSX) per pair

Notes
- File format is inferred from README and scripts/batch_export_ftms.py
- Data files may be unavailable at coding time; the script handles missing files gracefully.

Usage (after activating .venv311 or `conda activate van_krevelen_plotting`):
  python scripts/lost_gained_vankrevelen.py

Optional args:
  --data-dir /path/to/data/raw
  --out-dir  /path/to/output (default: data/raw/comparisons)
  --export-tables / --no-export-tables
  --pairs ICRMS-1_vs_ICRMS-2 ICRMS-1_vs_ICRMS-3 ...
  --label-lost 'Lost' --label-gained 'Gained' --label-shared 'Shared'
  --color-lost '#d62728' --color-gained '#2ca02c' --color-shared '#9e9e9e'
  --size-lost 12 --size-gained 12 --size-shared 8
  --title-template 'Lost & Gained Van Krevelen: {a} vs {b}'

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pykrev as pk
from pykrev.formula.msTuple import msTuple as PKmsTuple

# Try to import Spectrum (optional)
Spectrum = None
try:
    from pykrev.mass_spectrum import Spectrum as _Spectrum
    Spectrum = _Spectrum
except Exception:
    Spectrum = None

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'raw'
DEFAULT_OUT_DIR = ROOT / 'data' / 'comparisons'

# Default files and pairs
DEFAULT_FILES = [
    DATA_DIR / 'ICRMS-1.xlsx',
    DATA_DIR / 'ICRMS-2.xlsx',
    DATA_DIR / 'ICRMS-3.xlsx',
    DATA_DIR / 'ICRMS-4.xlsx',
]
DEFAULT_PAIRS = [
    ('ICRMS-1', 'ICRMS-2'),
    ('ICRMS-1', 'ICRMS-3'),
    ('ICRMS-1', 'ICRMS-4'),
    ('ICRMS-2', 'ICRMS-3'),
]

# Default display names for datasets
DISPLAY_NAME_MAP: Dict[str, str] = {
    'ICRMS-1': 'NOM-Initial',
    'ICRMS-2': 'NOM-Light',
    'ICRMS-3': 'NOM-CeO2-Light',
    'ICRMS-4': 'NOM-CeO2-Dark',
}


# -------------------------
# IO and utilities
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_icrms_xlsx(path: Path) -> pd.DataFrame:
    """Load the first sheet of an ICRMS Excel file and standardize columns.

    Expected columns in Excel: Formula (required); m/z, Intensity (optional)
    Returns a DataFrame with lowercase columns: at least 'formula'; includes 'mz', 'intensity' if present
    """
    xls = pd.ExcelFile(path)
    df = xls.parse(xls.sheet_names[0])
    rename_map = {
        'Formula': 'formula',
        'm/z': 'mz',
        'Intensity': 'intensity',
    }
    df = df.rename(columns=rename_map)
    if 'formula' not in df.columns:
        raise ValueError(f"Missing required column 'Formula' in {path}")

    # Keep needed columns and clean formulas
    keep_cols = [c for c in ['formula', 'mz', 'intensity'] if c in df.columns]
    df = df[keep_cols].copy()
    df['formula'] = (
        df['formula']
        .astype(str)
        .str.strip()
        .str.replace('\u200b', '', regex=False)  # remove zero-width if present
    )
    df = df[df['formula'] != '']

    # Deduplicate formulas within-file (keep first occurrence)
    df = df.drop_duplicates(subset='formula', keep='first')
    return df


# -------------------------
# msTuple helpers and VK coordinates via pykrev
# -------------------------

def to_ms_tuple(df: pd.DataFrame):
    """Build a pykrev msTuple from a DataFrame.

    - If 'mz' / 'intensity' exist, use them; otherwise use placeholders.
    - pykrev.msTuple expects (formula_list, intensity_array, mz_array)
    """
    formula_list = df['formula'].astype(str).tolist()
    n = len(formula_list)
    if 'intensity' in df.columns:
        intensity_array = df['intensity'].to_numpy()
    else:
        intensity_array = np.ones(n, dtype=float)
    if 'mz' in df.columns:
        mz_array = df['mz'].to_numpy()
    else:
        mz_array = np.zeros(n, dtype=float)
    return PKmsTuple(formula_list, intensity_array, mz_array)


def vk_coords_df_from_formulas(formulas: Iterable[str]) -> pd.DataFrame:
    """Compute VK coordinates (O/C, H/C) for a list of formulas using pykrev.

    Returns DataFrame with columns: formula, oc, hc
    """
    formulas = list(formulas)
    if not formulas:
        return pd.DataFrame(columns=['formula', 'oc', 'hc'])
    n = len(formulas)
    ms = PKmsTuple(formulas, np.ones(n, dtype=float), np.zeros(n, dtype=float))
    # Use pykrev to derive coordinates via its VK plot helper
    fig, ax = pk.van_krevelen_plot(ms, y_ratio='HC', c='density', s=1)
    scatter = ax.collections[0] if ax.collections else None
    oc = np.array([])
    hc = np.array([])
    if scatter is not None:
        offsets = np.asarray(scatter.get_offsets())
        if offsets.size:
            oc = offsets[:, 0]
            hc = offsets[:, 1]
    plt.close(fig)
    # Align lengths defensively
    m = min(len(formulas), len(oc))
    if m == 0:
        return pd.DataFrame(columns=['formula', 'oc', 'hc'])
    df = pd.DataFrame({'formula': formulas[:m], 'oc': oc[:m], 'hc': hc[:m]})
    # Drop non-finite
    df = df[np.isfinite(df['oc']) & np.isfinite(df['hc'])]
    return df


# -------------------------
# Comparison logic
# -------------------------

def categorize_formulas(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[set, set, set]:
    """Return (lost, gained, shared) sets of formulas for A vs B.

    - lost: in A only
    - gained: in B only
    - shared: in both
    """
    set_a = set(df_a['formula'].astype(str))
    set_b = set(df_b['formula'].astype(str))
    shared = set_a & set_b
    lost = set_a - set_b
    gained = set_b - set_a
    return lost, gained, shared


def global_axis_limits(dfs: List[pd.DataFrame]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute unified xlim (O/C) and ylim (H/C) via pykrev across provided DataFrames.

    Fallbacks: xlim=(0,1.2), ylim=(0,2.2)
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        # Build ms object (Spectrum preferred when available)
        ms_tuple = to_ms_tuple(df)
        ms_obj = None
        if Spectrum is not None:
            try:
                ms_obj = Spectrum(ms_tuple)
            except Exception:
                ms_obj = None
        use_obj = ms_obj if ms_obj is not None else ms_tuple
        try:
            fig, ax = pk.van_krevelen_plot(use_obj, y_ratio='HC', c='density', s=1)
            scatter = ax.collections[0] if ax.collections else None
            if scatter is not None:
                offsets = np.asarray(scatter.get_offsets())
                if offsets.size:
                    xs.append(offsets[:, 0])
                    ys.append(offsets[:, 1])
            plt.close(fig)
        except Exception:
            continue
    if xs and ys:
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        x_all = x_all[np.isfinite(x_all)]
        y_all = y_all[np.isfinite(y_all)]
        if x_all.size and y_all.size:
            x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
            xr = (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 0.1
            yr = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 0.1
            xlim = (max(0.0, x_min - 0.05 * xr), x_max + 0.05 * xr)
            ylim = (max(0.0, y_min - 0.05 * yr), y_max + 0.05 * yr)
        else:
            xlim = (0.0, 1.2)
            ylim = (0.0, 2.2)
    else:
        xlim = (0.0, 1.2)
        ylim = (0.0, 2.2)
    return xlim, ylim


# -------------------------
# Plotting
# -------------------------

def plot_lost_gained_vankrevelen(
    lost_df: pd.DataFrame,
    gained_df: pd.DataFrame,
    shared_df: pd.DataFrame,
    title: str,
    out_png: Path,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    label_lost: str = 'Lost',
    label_gained: str = 'Gained',
    label_shared: str = 'Shared',
    color_lost: str = '#d62728',
    color_gained: str = '#2ca02c',
    color_shared: str = '#9e9e9e',
    size_lost: float = 12,
    size_gained: float = 12,
    size_shared: float = 8,
):
    """Plot Lost, Gained, and Shared points on a Van Krevelen diagram.

    Parameters
    ---------
    lost_df, gained_df, shared_df : DataFrame with columns ['formula', 'oc', 'hc']
    title : str
    out_png : Path
    xlim, ylim : optional axis limits to enforce
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)

    # Plot shared as background
    if shared_df is not None and not shared_df.empty:
        ax.scatter(
            shared_df['oc'], shared_df['hc'],
            s=size_shared, c=color_shared, alpha=0.35,
            label=f'{label_shared} (n={len(shared_df)})', edgecolors='none'
        )

    # Plot lost (A only)
    if lost_df is not None and not lost_df.empty:
        ax.scatter(
            lost_df['oc'], lost_df['hc'],
            s=size_lost, c=color_lost, alpha=0.85,
            label=f'{label_lost} (n={len(lost_df)})', edgecolors='none'
        )

    # Plot gained (B only)
    if gained_df is not None and not gained_df.empty:
        ax.scatter(
            gained_df['oc'], gained_df['hc'],
            s=size_gained, c=color_gained, alpha=0.85,
            label=f'{label_gained} (n={len(gained_df)})', edgecolors='none'
        )

    ax.set_xlabel('O/C')
    ax.set_ylabel('H/C')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.grid(False)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# -------------------------
# Venn diagram (2-set) with area matching
# -------------------------

def _circle_overlap_area(r1: float, r2: float, d: float) -> float:
    """Area of overlap between two circles of radii r1, r2 separated by distance d."""
    import math
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    # partial overlap
    r1_sq, r2_sq = r1 * r1, r2 * r2
    alpha = math.acos((d * d + r1_sq - r2_sq) / (2.0 * d * r1))
    beta = math.acos((d * d + r2_sq - r1_sq) / (2.0 * d * r2))
    return r1_sq * alpha + r2_sq * beta - d * r1 * math.sin(alpha)


def _solve_center_distance_for_overlap(r1: float, r2: float, target_overlap: float) -> float:
    """Solve for center distance d so that circle overlap area equals target_overlap."""
    import math
    eps = 1e-10
    max_overlap = math.pi * min(r1, r2) ** 2
    if target_overlap <= 0:
        return r1 + r2
    if target_overlap >= max_overlap:
        return abs(r1 - r2)
    lo, hi = abs(r1 - r2), r1 + r2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        area = _circle_overlap_area(r1, r2, mid)
        if abs(area - target_overlap) < 1e-6:
            return mid
        if area > target_overlap:
            lo = mid
        else:
            hi = mid
        if hi - lo < eps:
            break
    return 0.5 * (lo + hi)


def make_venn_diagram(
    name_a: str,
    name_b: str,
    n_a_only: int,
    n_b_only: int,
    n_shared: int,
    out_png: Path,
    fill_a: str = '#C9D2DC',
    edge_a: str = '#9AA7B6',
    fill_b: str = '#AFC8E3',
    edge_b: str = '#86AED4',
    fill_inter: str = '#9FB3C7',
    alpha_fill: float = 0.75,
    alpha_edge: float = 0.9,
    lw: float = 1.8,
    title: str | None = None,
):
    import math
    total_a = float(n_a_only + n_shared)
    total_b = float(n_b_only + n_shared)
    # percentages of shared in each set
    perc_shared_a = (n_shared / total_a) if total_a > 0 else 0.0
    perc_shared_b = (n_shared / total_b) if total_b > 0 else 0.0

    # Radii so that areas are proportional to counts (area units = counts)
    r1 = math.sqrt(total_a / math.pi)
    r2 = math.sqrt(total_b / math.pi)
    target_overlap = float(n_shared)
    d = _solve_center_distance_for_overlap(r1, r2, target_overlap)

    # Place centers on x-axis at (-d/2, 0) and (d/2, 0)
    cx_a, cy_a = -0.5 * d, 0.0
    cx_b, cy_b = 0.5 * d, 0.0

    # Build figure
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    ax.set_aspect('equal')

    # Base circles (full fills)
    circ_a = Circle((cx_a, cy_a), r1, facecolor=fill_a, edgecolor=edge_a, lw=lw, alpha=alpha_fill)
    circ_b = Circle((cx_b, cy_b), r2, facecolor=fill_b, edgecolor=edge_b, lw=lw, alpha=alpha_fill)
    ax.add_patch(circ_a)
    ax.add_patch(circ_b)

    # Edge overlays (stronger edge alpha without affecting fill)
    edgeA = Circle((cx_a, cy_a), r1, facecolor='none', edgecolor=edge_a, lw=lw, alpha=alpha_edge, zorder=4)
    edgeB = Circle((cx_b, cy_b), r2, facecolor='none', edgecolor=edge_b, lw=lw, alpha=alpha_edge, zorder=4)
    ax.add_patch(edgeA)
    ax.add_patch(edgeB)

    # Intersection overlay: draw right circle clipped by left circle
    inter = Circle((cx_b, cy_b), r2, facecolor=fill_inter, edgecolor='none', alpha=min(1.0, alpha_fill + 0.15), zorder=3)
    inter.set_clip_path(circ_a)
    ax.add_patch(inter)

    # (remove circle-top labels to keep only two aligned legend rows at the top)

    # Counts (use neutral dark text for contrast)
    txt_color = '#222222'
    ax.text(cx_a - r1 * 0.38, cy_a, f"{int(n_a_only)}", ha='center', va='center', fontsize=12, color=txt_color)
    ax.text(cx_b + r2 * 0.38, cy_b, f"{int(n_b_only)}", ha='center', va='center', fontsize=12, color=txt_color)
    ax.text(0.0, 0.0, f"{int(n_shared)}", ha='center', va='center', fontsize=12, color=txt_color)

    # Percentages by region relative to union total
    union_total = float(n_a_only + n_b_only + n_shared)
    p_left = (n_a_only / union_total) if union_total > 0 else 0.0
    p_shared = (n_shared / union_total) if union_total > 0 else 0.0
    p_right = (n_b_only / union_total) if union_total > 0 else 0.0

    # Place percentages near their respective regions
    ax.text(cx_a - r1 * 0.38, cy_a - r1 * 0.42, f"({int(round(p_left*100))}%)",
            ha='center', va='center', fontsize=10, color=txt_color)
    ax.text(0.0, -0.45 * min(r1, r2), f"({int(round(p_shared*100))}%)",
            ha='center', va='center', fontsize=10, color=txt_color)
    ax.text(cx_b + r2 * 0.38, cy_b - r2 * 0.42, f"({int(round(p_right*100))}%)",
            ha='center', va='center', fontsize=10, color=txt_color)

    # Limits with padding
    min_x = min(cx_a - r1, cx_b - r2)
    max_x = max(cx_a + r1, cx_b + r2)
    min_y = min(cy_a - r1, cy_b - r2)
    max_y = max(cy_a + r1, cy_b + r2)
    pad_x = 0.12 * (max_x - min_x)
    pad_y = 0.15 * (max_y - min_y)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.axis('off')

    # Two aligned legend rows at the very top (no title)
    leg_box = 0.045
    x0 = 0.06
    y1 = 0.975
    y2 = 0.915
    # Row 1: left-region legend
    ax.add_patch(Rectangle((x0, y1 - leg_box/2), leg_box, leg_box, transform=ax.transAxes,
                           facecolor=fill_a, edgecolor=edge_a, lw=1.2, alpha=1.0, zorder=5))
    ax.text(x0 + leg_box + 0.01, y1, name_a, transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color='#222222')
    # Row 2: right-region legend
    ax.add_patch(Rectangle((x0, y2 - leg_box/2), leg_box, leg_box, transform=ax.transAxes,
                           facecolor=fill_b, edgecolor=edge_b, lw=1.2, alpha=1.0, zorder=5))
    ax.text(x0 + leg_box + 0.01, y2, name_b, transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color='#222222')
    plt.tight_layout()

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# Orchestration
# -------------------------

def compare_pair(
    name_a: str,
    name_b: str,
    df_map: Dict[str, pd.DataFrame],
    out_root: Path,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    export_tables: bool = True,
    label_lost: str = 'Lost',
    label_gained: str = 'Gained',
    label_shared: str = 'Shared',
    color_lost: str = '#d62728',
    color_gained: str = '#2ca02c',
    color_shared: str = '#9e9e9e',
    size_lost: float = 12,
    size_gained: float = 12,
    size_shared: float = 8,
    title_template: str = 'Lost & Gained Van Krevelen: {a} vs {b}',
    venn_color_a: str = '#9e9e9e',
    venn_color_b: str = '#1f77b4',
    venn_title_template: str = 'Venn: {a} vs {b}',
):
    """Produce plot and optional tables for a pair (A,B)."""
    if name_a not in df_map or name_b not in df_map:
        print(f"[WARN] Skipping pair {name_a} vs {name_b}: missing data")
        return

    df_a = df_map[name_a]
    df_b = df_map[name_b]
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        print(f"[WARN] Skipping pair {name_a} vs {name_b}: empty data")
        return

    lost, gained, shared = categorize_formulas(df_a, df_b)
    lost_df = vk_coords_df_from_formulas(lost)
    gained_df = vk_coords_df_from_formulas(gained)
    shared_df = vk_coords_df_from_formulas(shared)

    pair_dir = out_root / f'{name_a}_vs_{name_b}'
    ensure_dir(pair_dir)

    # Plot (use display names in title template)
    disp_a_title = DISPLAY_NAME_MAP.get(name_a, name_a)
    disp_b_title = DISPLAY_NAME_MAP.get(name_b, name_b)
    title = title_template.format(a=disp_a_title, b=disp_b_title)
    out_png = pair_dir / 'van_krevelen_lost_gained.png'
    plot_lost_gained_vankrevelen(
        lost_df, gained_df, shared_df, title, out_png,
        xlim=xlim, ylim=ylim,
        label_lost=label_lost, label_gained=label_gained, label_shared=label_shared,
        color_lost=color_lost, color_gained=color_gained, color_shared=color_shared,
        size_lost=size_lost, size_gained=size_gained, size_shared=size_shared,
    )
    print(f'[INFO] Wrote figure: {out_png}')

    # Export tables (optional)
    if export_tables:
        out_xlsx = pair_dir / 'lost_gained_shared.xlsx'
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            lost_df.sort_values(['oc', 'hc']).to_excel(w, index=False, sheet_name='Lost (A only)')
            gained_df.sort_values(['oc', 'hc']).to_excel(w, index=False, sheet_name='Gained (B only)')
            shared_df.sort_values(['oc', 'hc']).to_excel(w, index=False, sheet_name='Shared (A ∩ B)')
        print(f'[INFO] Wrote tables: {out_xlsx}')

    # Venn diagram
    venn_png = pair_dir / 'venn.png'
    n_a_only = len(lost)
    n_b_only = len(gained)
    n_shared = len(shared)
    # Use display names for legends at the top
    disp_a = DISPLAY_NAME_MAP.get(name_a, name_a)
    disp_b = DISPLAY_NAME_MAP.get(name_b, name_b)
    venn_title = venn_title_template
    make_venn_diagram(
        disp_a, disp_b, n_a_only, n_b_only, n_shared, venn_png,
        fill_a=venn_color_a, fill_b=venn_color_b, title=venn_title,
    )
    print(f'[INFO] Wrote Venn: {venn_png}')


def parse_pair_strings(pair_strs: List[str]) -> List[Tuple[str, str]]:
    """Parse pair strings like 'ICRMS-1_vs_ICRMS-2' into tuples (ICRMS-1, ICRMS-2)."""
    pairs: List[Tuple[str, str]] = []
    for s in pair_strs:
        if '_vs_' in s:
            a, b = s.split('_vs_', 1)
            pairs.append((a, b))
    return pairs


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description='Lost & Gained Van Krevelen plots for ICRMS pairs')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR), help='Directory containing ICRMS-#.xlsx files (default: data/raw)')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR), help='Directory for outputs (figures and tables), default: data/comparisons')
    parser.add_argument('--export-tables', action=argparse.BooleanOptionalAction, default=True, help='Export category tables (XLSX) per pair')
    parser.add_argument('--pairs', nargs='*', default=[f'{a}_vs_{b}' for a,b in DEFAULT_PAIRS], help='Pairs to compare, e.g. ICRMS-1_vs_ICRMS-2')
    # Customization options
    parser.add_argument('--label-lost', type=str, default='Lost', help='Legend label for Lost')
    parser.add_argument('--label-gained', type=str, default='Gained', help='Legend label for Gained')
    parser.add_argument('--label-shared', type=str, default='Shared', help='Legend label for Shared')
    parser.add_argument('--color-lost', type=str, default='#d62728', help='Color for Lost points')
    parser.add_argument('--color-gained', type=str, default='#2ca02c', help='Color for Gained points')
    parser.add_argument('--color-shared', type=str, default='#9e9e9e', help='Color for Shared points')
    parser.add_argument('--size-lost', type=float, default=12, help='Marker size for Lost points')
    parser.add_argument('--size-gained', type=float, default=12, help='Marker size for Gained points')
    parser.add_argument('--size-shared', type=float, default=8, help='Marker size for Shared points')
    parser.add_argument('--title-template', type=str, default='Lost & Gained Van Krevelen: {a} vs {b}', help='Title template; available fields: {a}, {b}')
    # Venn customization
    parser.add_argument('--venn-color-a', type=str, default='#FBE6C1', help='Venn fill color for dataset A')
    parser.add_argument('--venn-color-b', type=str, default='#ADB8DF', help='Venn fill color for dataset B')
    parser.add_argument('--venn-title-template', type=str, default='Venn: {a} vs {b}', help='Venn title template; fields: {a}, {b}')

    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Map dataset name -> DataFrame
    wanted_names = set()
    for s in args.pairs:
        parts = s.split('_vs_')
        if len(parts) == 2:
            wanted_names.add(parts[0])
            wanted_names.add(parts[1])
    # Also include defaults if present on disk
    for p in DEFAULT_FILES:
        if p.exists():
            wanted_names.add(p.stem)

    df_map: Dict[str, pd.DataFrame] = {}
    for name in sorted(wanted_names):
        xlsx_path = data_dir / f'{name}.xlsx'
        if not xlsx_path.exists():
            print(f'[WARN] Missing file: {xlsx_path}')
            continue
        try:
            df = load_icrms_xlsx(xlsx_path)
            df_map[name] = df
            print(f'[INFO] Loaded {name}: {len(df)} unique formulas')
        except Exception as e:
            print(f'[WARN] Failed to load {xlsx_path}: {e}')

    if not df_map:
        print('[ERROR] No input data loaded. Nothing to do.')
        return 1

    # Compute unified axis limits
    xlim, ylim = global_axis_limits(list(df_map.values()))
    print(f'[INFO] Global axis limits -> xlim: {xlim}, ylim: {ylim}')

    # Run comparisons
    pairs = parse_pair_strings(args.pairs)
    if not pairs:
        pairs = DEFAULT_PAIRS
    for a, b in pairs:
        disp_a = DISPLAY_NAME_MAP.get(a, a)
        disp_b = DISPLAY_NAME_MAP.get(b, b)
        compare_pair(
            a, b, df_map, out_dir, xlim, ylim,
            export_tables=args.export_tables,
            label_lost=args.label_lost,
            label_gained=args.label_gained,
            label_shared=args.label_shared,
            color_lost=args.color_lost,
            color_gained=args.color_gained,
            color_shared=args.color_shared,
            size_lost=args.size_lost,
            size_gained=args.size_gained,
            size_shared=args.size_shared,
            title_template=args.title_template.replace('{a}', disp_a).replace('{b}', disp_b),
            venn_color_a=args.venn_color_a,
            venn_color_b=args.venn_color_b,
            venn_title_template=args.venn_title_template.replace('{a}', disp_a).replace('{b}', disp_b),
        )

    print('[INFO] Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
