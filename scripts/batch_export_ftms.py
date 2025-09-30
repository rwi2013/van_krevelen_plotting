#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch export for FT-MS data (ICRMS-1..4):
- Load Excel from data/ICRMS-#.xlsx (first sheet), standardize columns to formula/mz/intensity
- Build msTuple (and Spectrum if available)
- Compute DBE, massExpected (monoisotopic, ion_charge=-1, protonated=True), and massError (ppm)
- Plot Van Krevelen (DBE colored) and save as PNG
- Unify Van Krevelen y-axis and colorbar ranges (DBE & density) across all datasets for consistent figures
- Save augmented table (CSV and XLSX) into per-file output directory
- Zip the output directory
"""
from __future__ import annotations
import os
from pathlib import Path
import types
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 构造一个假模块 numpy.lib.function_base，只提供 pykrev 需要的对象
# mod = types.ModuleType('numpy.lib.function_base')
# mod.rot90 = np.rot90
# mod._rot90_dispatcher = lambda m, k=1, axes=(0,1): (m,)  # 占位即可
# sys.modules['numpy.lib.function_base'] = mod
# import pykrev as pk

import pykrev as pk
from pykrev.formula.msTuple import msTuple as PKmsTuple

# Try to import Spectrum (optional)
Spectrum = None
try:
    from pykrev.mass_spectrum import Spectrum as _Spectrum
    Spectrum = _Spectrum
except Exception:
    Spectrum = None

ROOT = Path(__file__).resolve().parent.parent  # project root (/home/...)
DATA_DIR = ROOT / 'data' / 'raw'  # read inputs from data/raw
OUT_ROOT = ROOT / 'data'          # write outputs under data/

FILES = [
    DATA_DIR / 'ICRMS-1.xlsx',
    DATA_DIR / 'ICRMS-2.xlsx',
    DATA_DIR / 'ICRMS-3.xlsx',
    DATA_DIR / 'ICRMS-4.xlsx',
]


def load_icrms_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    df = xls.parse(xls.sheet_names[0])
    rename_map = {
        'Formula': 'formula',
        'm/z': 'mz',
        'Intensity': 'intensity',
    }
    df = df.rename(columns=rename_map)
    missing = [c for c in ['formula', 'mz', 'intensity'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns {missing} in {path}')
    df = df[['formula','mz','intensity']].copy()
    return df


def to_ms_tuple(df: pd.DataFrame):
    formula_list = df['formula'].astype(str).tolist()
    mz_array = df['mz'].to_numpy()
    intensity_array = df['intensity'].to_numpy()
    # pykrev.msTuple expects (formula, intensity, mz)
    return PKmsTuple(formula_list, intensity_array, mz_array)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def zip_dir(folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = Path(root) / fn
                zf.write(fp, arcname=str(fp.relative_to(folder.parent)))


def compute_global_limits(file_paths):
    """Pre-scan all datasets to get unified limits for plotting.

    Returns
    -------
    y_lims : tuple(float,float)
        Global y-axis limits for Van Krevelen (H/C).
    dbe_limits : tuple(float,float)
        Global color scale limits for DBE.
    density_limits : tuple(float,float)
        Global color scale limits for density-colored VK plot.
    """
    y_min, y_max = np.inf, -np.inf
    dbe_min, dbe_max = np.inf, -np.inf
    dens_min, dens_max = np.inf, -np.inf

    for xlsx_path in file_paths:
        if not Path(xlsx_path).exists():
            print(f"[WARN] Missing file during pre-scan: {xlsx_path}")
            continue
        try:
            df = load_icrms_xlsx(Path(xlsx_path))
        except Exception as e:
            print(f"[WARN] Pre-scan load failed for {xlsx_path}: {e}")
            continue

        msTuple = to_ms_tuple(df)
        ms_obj = None
        if Spectrum is not None:
            try:
                ms_obj = Spectrum(msTuple)
            except Exception:
                ms_obj = None
        use_obj = ms_obj if ms_obj is not None else msTuple

        # DBE limits
        try:
            dbe_vals = np.asarray(pk.double_bond_equivalent(use_obj), dtype=float)
            dbe_vals = dbe_vals[np.isfinite(dbe_vals)]
            if dbe_vals.size:
                dbe_min = min(dbe_min, float(np.min(dbe_vals)))
                dbe_max = max(dbe_max, float(np.max(dbe_vals)))
        except Exception as e:
            print('[WARN] Pre-scan DBE failed:', e)

        # Density limits and y-axis (H/C) range via temporary VK plot
        try:
            fig, ax = pk.van_krevelen_plot(use_obj, y_ratio='HC', c='density', s=7)
            scatter = ax.collections[0] if ax.collections else None
            if scatter is not None:
                # y from offsets
                try:
                    offsets = scatter.get_offsets()
                    pts = np.asarray(offsets)
                    if pts.size:
                        ys = pts[:, 1]
                        ys = ys[np.isfinite(ys)]
                        if ys.size:
                            y_min = min(y_min, float(np.min(ys)))
                            y_max = max(y_max, float(np.max(ys)))
                except Exception:
                    pass
                # density range from color array
                try:
                    arr = np.asarray(scatter.get_array())
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        dens_min = min(dens_min, float(np.min(arr)))
                        dens_max = max(dens_max, float(np.max(arr)))
                except Exception:
                    pass
            plt.close(fig)
        except Exception as e:
            print('[WARN] Pre-scan density VK failed:', e)

    # Fallbacks if any are still inf/-inf
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
        y_lims = (0.0, 2.0)
    else:
        y_lims = (y_min, y_max)

    if not np.isfinite(dbe_min) or not np.isfinite(dbe_max) or dbe_min >= dbe_max:
        dbe_limits = (0.0, 30.0)
    else:
        dbe_limits = (dbe_min, dbe_max)

    if not np.isfinite(dens_min) or not np.isfinite(dens_max) or dens_min >= dens_max:
        density_limits = (0.0, 1.0)
    else:
        density_limits = (dens_min, dens_max)

    print('[INFO] Global limits:', 'y', y_lims, 'DBE', dbe_limits, 'density', density_limits)
    return y_lims, dbe_limits, density_limits


def make_van_krevelen(ms_obj, dbe, out_png: Path, y_lims=None, dbe_limits=None):
    fig, ax = pk.van_krevelen_plot(ms_obj, y_ratio='HC', c=dbe, s=7, cmap='plasma')
    scatter = ax.collections[0] if ax.collections else None
    # Apply unified color scale for DBE if provided
    if scatter is not None and dbe_limits is not None:
        try:
            scatter.set_clim(dbe_limits[0], dbe_limits[1])
        except Exception:
            pass
    # Apply unified y-axis limits if provided
    if y_lims is not None:
        ax.set_ylim(y_lims)
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('DBE')
    ax.set_title('Van Krevelen Plot (colored by DBE)')
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def make_extra_plots(ms_obj, dbe, ai, out_dir: Path, y_lims=None, density_limits=None):
    """Generate additional plots and save to out_dir."""
    # 5A. Density-colored Van Krevelen
    try:
        fig, ax = pk.van_krevelen_plot(ms_obj, y_ratio='HC', c='density', s=7)
        scatter = ax.collections[0] if ax.collections else None
        # Apply unified color scale for density if provided
        if scatter is not None and density_limits is not None:
            try:
                scatter.set_clim(density_limits[0], density_limits[1])
            except Exception:
                pass
        # Apply unified y-axis limits if provided
        if y_lims is not None:
            ax.set_ylim(y_lims)
        plt.colorbar(scatter if scatter is not None else ax.collections[0], ax=ax).set_label('Kernel Density')
        ax.grid(False)
        plt.tight_layout()
        fig.savefig(out_dir / 'van_krevelen_density.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] density VK failed:', e)

    # 5B. Histogram heatmap (equal bins)
    try:
        fig, ax, d_index = pk.van_krevelen_histogram(ms_obj, bins=[10, 10], cmap='viridis')
        plt.colorbar(ax.collections[0], ax=ax).set_label('Counts')
        plt.tight_layout()
        fig.savefig(out_dir / 'van_krevelen_hist_10x10.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] VK histogram 10x10 failed:', e)

    # 5C. Histogram heatmap (custom bins)
    try:
        fig, ax, d_index = pk.van_krevelen_histogram(
            ms_obj,
            bins=[np.linspace(0, 1, 5), np.linspace(0, 2, 5)],
            cmap='cividis'
        )
        plt.colorbar(ax.collections[0], ax=ax).set_label('Counts')
        plt.tight_layout()
        fig.savefig(out_dir / 'van_krevelen_hist_custom.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] VK histogram custom failed:', e)

    # Kendrick mass defect plot (colored by AI)
    try:
        fig, ax, (_km, _kmd) = pk.kendrick_mass_defect_plot(
            ms_obj, base='CHON', rounding='ceil', s=3, c=ai
        )
        plt.colorbar(ax.collections[0], ax=ax).set_label('Aromaticity Index')
        plt.xlim([0, 1000])
        plt.tight_layout()
        fig.savefig(out_dir / 'kendrick_mass_defect_ai.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] KMD plot failed:', e)

    # Atomic class plot (element O)
    try:
        fig, ax, (_mean, _median, _sigma) = pk.atomic_class_plot(
            ms_obj, element='O', color='b', summary_statistics=True, bins=range(0, 33)
        )
        plt.tight_layout()
        fig.savefig(out_dir / 'atomic_class_O.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] atomic_class_plot failed:', e)

    # Compound class plot (MSCC)
    try:
        fig, ax, (_compounds, _counts) = pk.compound_class_plot(ms_obj, color='g', method='MSCC')
        plt.tight_layout()
        fig.savefig(out_dir / 'compound_class_MSCC.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] compound_class_plot failed:', e)

    # Mass histograms
    try:
        fig, ax, (_mean, _median, _sigma) = pk.mass_histogram(
            ms_obj,
            method='monoisotopic',
            bin_width=20,
            summary_statistics=True,
            color='blue',
            alpha=0.5,
            kde=True,
            kde_color='blue',
            density=False,
        )
        plt.xlabel('Monoisotopic atomic mass (Da)')
        plt.tight_layout()
        fig.savefig(out_dir / 'mass_histogram_mono_kde.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] mass_histogram mono+kde failed:', e)

    try:
        fig, ax, (_mean, _median, _sigma) = pk.mass_histogram(
            ms_obj,
            method='me',  # as per reference snippet
            kde=True,
            hist=False,
            kde_color='red',
            summary_statistics=True,
            deprotonated=True,
        )
        plt.tight_layout()
        fig.savefig(out_dir / 'mass_histogram_me_kde_only.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] mass_histogram me+kde failed:', e)

    # Mass spectrum (measured m/z)
    try:
        res = pk.mass_spectrum(
            ms_obj,
            method='mz',
            logTransform=False,
            stepSize=4,
            lineColor='g',
            lineWidth=1.2,
        )
        # mass_spectrum returns (fig, ax1) unless invertedAxis is used; handle both
        if isinstance(res, tuple) and len(res) == 2:
            fig, ax1 = res
        else:
            fig, ax1, _ = res
        plt.tight_layout()
        fig.savefig(out_dir / 'mass_spectrum_mz.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print('[WARN] mass_spectrum (mz) failed:', e)

def process_one(xlsx_path: Path, y_lims=None, dbe_limits=None, density_limits=None):
    if not xlsx_path.exists():
        print(f'[WARN] Missing file: {xlsx_path}')
        return

    name = xlsx_path.stem  # e.g., ICRMS-1
    out_dir = OUT_ROOT / f'{name}_output'
    ensure_dir(out_dir)

    print(f'== Processing {xlsx_path.name} -> {out_dir} ==')
    df = load_icrms_xlsx(xlsx_path)

    msTuple = to_ms_tuple(df)
    ms_obj = None
    if Spectrum is not None:
        try:
            ms_obj = Spectrum(msTuple)
        except Exception:
            ms_obj = None
    use_obj = ms_obj if ms_obj is not None else msTuple

    # Core calculations
    dbe = pk.double_bond_equivalent(use_obj)
    ai = None
    try:
        ai = pk.aromaticity_index(use_obj, index_type='rAI')
    except Exception as e:
        print('[WARN] aromaticity_index failed:', e)
    massExpected = pk.calculate_mass(use_obj, method='monoisotopic', ion_charge=-1, protonated=True)
    massError = (massExpected - df['mz'].to_numpy()) / df['mz'].to_numpy() * 1e6

    # Augment table
    df_out = df.copy()
    df_out['massExpected'] = massExpected
    df_out['massError'] = massError

    # Plots
    vk_png = out_dir / 'van_krevelen_dbe.png'
    make_van_krevelen(use_obj, dbe, vk_png, y_lims=y_lims, dbe_limits=dbe_limits)
    make_extra_plots(use_obj, dbe, ai, out_dir, y_lims=y_lims, density_limits=density_limits)

    # Save tables
    csv_path = out_dir / f'{name}_augmented.csv'
    xlsx_out = out_dir / f'{name}_augmented.xlsx'
    df_out.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_out, engine='openpyxl') as w:
        df_out.to_excel(w, index=False, sheet_name='Sheet1')

    # Zip folder
    zip_path = OUT_ROOT / f'{name}_output.zip'
    zip_dir(out_dir, zip_path)

    print(f'Wrote: {csv_path}')
    print(f'Wrote: {xlsx_out}')
    print(f'Wrote: {vk_png}')
    print(f'Zipped: {zip_path}')


def main():
    # Pre-scan to get unified plotting ranges
    y_lims, dbe_limits, density_limits = compute_global_limits(FILES)
    for p in FILES:
        process_one(p, y_lims=y_lims, dbe_limits=dbe_limits, density_limits=density_limits)


if __name__ == '__main__':
    main()
