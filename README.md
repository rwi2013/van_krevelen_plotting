# FT-MS Data Processing Tools

This repo contains two main entry points and a minimal workflow. Use this quick guide and skip the rest unless you need details.

## Quick Start (TL;DR)

```bash
# 1) Activate environment
conda activate van_krevelen_plotting

# 2) Paths
#   Inputs   : data/raw/ICRMS-#.xlsx
#   Batch out: data/<ICRMS-#>_output/ (+ zip under data/)
#   Compare  : data/comparisons/<A>_vs_<B>/

# 3) Batch export per file (tables + figures)
python scripts/batch_export_ftms.py

# 4) Pairwise Lost/Gained + Venn (area-matched)
python scripts/lost_gained_vankrevelen.py

# 5) Generate 2x2 grid plots (optional)
python scripts/plot_density_grid.py          # Density-colored VK plots
python scripts/plot_lost_gained_grid.py      # Lost/Gained comparison grid
```

Defaults:
- Display names: ICRMS-1→NOM-Initial, ICRMS-2→NOM-Light, ICRMS-3→NOM-CeO2-Light, ICRMS-4→NOM-CeO2-Dark
- Venn colors: left `#FBE6C1`, right `#ADB8DF`
- Pair outputs under `data/comparisons/`

Main parts of this code were informed by the following resources on preliminary FT-MS data processing:

- [FT-MS质谱数据的初步处理（文章）](https://mp.weixin.qq.com/s/UtmS3W1_LCmBhhLTyFFxPg)
- [FT-MS质谱数据的初步处理（视频）](https://www.bilibili.com/video/BV1SMomYwEs8)
Accessed on.

## Project Structure

```
├── scripts/
│   ├── FT-MS质谱数据的初步处理.ipynb  # Interactive analysis notebook
│   ├── batch_export_ftms.py           # Batch processing: tables + individual plots
│   ├── lost_gained_vankrevelen.py     # Lost/Gained + Venn (area-matched)
│   ├── plot_density_grid.py           # 2x2 grid: density-colored VK plots
│   └── plot_lost_gained_grid.py       # 2x2 grid: Lost/Gained VK comparisons
├── data/
│   ├── raw/                           # Put input Excel here (ICRMS-#.xlsx)
│   ├── ICRMS-#_output/                # Batch processing output (auto-generated)
│   ├── density_grid/                  # Density grid plots (auto-generated)
│   └── comparisons_grid/              # Lost/Gained grid plots (auto-generated)
├── .venv/                             # Python 3.12 virtual environment
├── .venv311/                          # Python 3.11 virtual environment (for pykrev)
└── README.md
```

### Script Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `batch_export_ftms.py` | Process individual datasets | `data/ICRMS-#_output/` |
| `lost_gained_vankrevelen.py` | Pairwise Lost/Gained comparison | `data/raw/comparisons/` |
| `plot_density_grid.py` | 2x2 grid of density plots | `data/density_grid/` |
| `plot_lost_gained_grid.py` | 2x2 grid of Lost/Gained plots | `data/comparisons_grid/` |

**Note**: Alternative styling options from previous versions are preserved as comments in each script for easy customization.

## Features

- **Molecular Formula Analysis**: Calculate element counts and ratios (H/C, O/H, O/C)
- **Chemical Indices**: Double Bond Equivalence (DBE), Aromaticity Index (rAI), Nominal Oxidation State of Carbon (NOSC)
- **Mass Calculations**: Nominal, average, monoisotopic, and theoretical masses
- **Mass Error**: Calculate relative mass error between measured and theoretical masses (ppm)
- **Kendrick Analysis**: Kendrick mass and mass defect calculations
- **Data Visualization**: Van Krevelen plots, mass spectra, atomic class distributions, etc.
- **Batch Processing**: Automated processing of multiple files with result export

## Data Requirements

Input files should be Excel format (.xlsx) containing the following required columns:
- `Formula`: Molecular formula (e.g., C6H12O6)
- `m/z`: Mass-to-charge ratio
- `Intensity`: Peak intensity

## Environment Setup (brief)

### Important Note: Dependency Version Conflicts

This project uses two different Python environments to resolve compatibility issues between `pykrev` and NumPy 2.0+:

1. **`.venv` (Python 3.12)**: For running Jupyter Notebooks
   - Contains NumPy 2.0+ compatibility patches
   - Suitable for interactive analysis

2. **`.venv311` (Python 3.11)**: For running batch processing scripts
   - Uses NumPy 1.26.4 to ensure full compatibility with pykrev 1.2.3
   - Suitable for production batch processing

### Environment 1: Jupyter Notebook Environment (Python 3.12)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install jupyter ipykernel numpy pandas matplotlib openpyxl pykrev scipy networkx>=2

# Register Jupyter kernel
python -m ipykernel install --user --name=ftms-env --display-name "Python (.venv) FT-MS"
```

### Environment 2: Batch Processing Environment (Python 3.11)

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install fixed version dependencies (key: numpy<2)
pip install --upgrade pip
pip install "numpy==1.26.4" "pykrev==1.2.3" pandas matplotlib openpyxl scipy networkx
```

## Usage

### Interactive Analysis

```bash
# Activate Jupyter environment
source .venv/bin/activate
jupyter notebook

# Open scripts/FT-MS质谱数据的初步处理.ipynb
# Select kernel: Python (.venv) FT-MS
```

### Batch Processing

```bash
# Activate batch processing environment
source .venv311/bin/activate

# Run batch processing script
python scripts/batch_export_ftms.py
```

Batch processing generates for each input file:
- Enhanced data tables (CSV and Excel formats)
- Van Krevelen plots and other visualization charts
- Compressed output folders

### Lost & Gained Van Krevelen Comparison (Python 3.11)

```bash
# Activate batch processing environment
source .venv311/bin/activate
# or conda activate van_krevelen_plotting

# Run comparison script (outputs to data/raw/comparisons/)
python scripts/lost_gained_vankrevelen.py

# Optional arguments
# --data-dir         Directory containing ICRMS-#.xlsx (default: data/raw)
# --out-dir          Output directory for figures and tables (default: data/raw/comparisons)
# --no-export-tables Disable exporting lost/gained/shared tables
# --pairs            Specify pairs, e.g. ICRMS-1_vs_ICRMS-2 ICRMS-1_vs_ICRMS-3 ...
# --label-lost       Legend label for Lost (default: 'Lost')
# --label-gained     Legend label for Gained (default: 'Gained')
# --label-shared     Legend label for Shared (default: 'Shared')
# --color-lost       Color for Lost points (default: '#d62728')
# --color-gained     Color for Gained points (default: '#2ca02c')
# --color-shared     Color for Shared points (default: '#9e9e9e')
# --size-lost        Marker size for Lost (default: 12)
# --size-gained      Marker size for Gained (default: 12)
# --size-shared      Marker size for Shared (default: 8)
# --title-template   Title template with {a}/{b} placeholders
# --venn-color-a     Venn fill color for dataset A (default: '#FBE6C1')
# --venn-color-b     Venn fill color for dataset B (default: '#ADB8DF')
# --venn-title-template  (Venn title is currently not shown; kept for compatibility)
```

Pairs generated by default:
- ICRMS-1 vs ICRMS-2
- ICRMS-1 vs ICRMS-3
- ICRMS-1 vs ICRMS-4
- ICRMS-2 vs ICRMS-3

The script reads Excel files from `data/raw/`, computes Lost (A\B), Gained (B\A), and Shared (A∩B) strictly by the
`Formula` column (no intensity/mz thresholds), uses `pykrev` to obtain Van Krevelen coordinates (O/C, H/C), and plots
these categories on a single diagram under unified X/Y axis ranges across all datasets. For each pair, a Venn diagram
with area matched to counts is also produced.

#### Customize legend, colors, and title

Examples:

```bash
# Custom legend text and colors
python scripts/lost_gained_vankrevelen.py \
  --label-lost "Disappeared" --label-gained "Formed" --label-shared "Background" \
  --color-lost "#e41a1c" --color-gained "#4daf4a" --color-shared "#999999"

# Adjust marker sizes and title
python scripts/lost_gained_vankrevelen.py \
  --size-lost 14 --size-gained 14 --size-shared 7 \
  --title-template "Lost & Gained VK: {a} vs {b}"

# Customize Venn diagram colors
python scripts/lost_gained_vankrevelen.py \
  --venn-color-a "#FBE6C1" --venn-color-b "#ADB8DF"

```

#### Display names and legends

Default display names are used in titles and Venn legends:

```
ICRMS-1 -> NOM-Initial
ICRMS-2 -> NOM-Light
ICRMS-3 -> NOM-CeO2-Light
ICRMS-4 -> NOM-CeO2-Dark
```

Venn diagram legends appear as two aligned rows at the top (left-row for the left dataset, second-row for the right dataset),
no Venn title. Percentages shown in the Venn are left-only/shared/right-only as fractions of the union.
```

### 2x2 Grid Plots

Generate publication-ready multi-panel figures with consistent styling across all datasets:

#### Density Grid Plot

```bash
# Activate batch processing environment
source .venv311/bin/activate

# Generate 2x2 grid of density-colored Van Krevelen plots
python scripts/plot_density_grid.py
```

Output: `data/density_grid/density_grid.png`

Features:
- Unified axis limits across all 4 datasets
- Gaussian kernel density estimation with `scipy.stats.gaussian_kde`
- Customizable color maps, point sizes, and transparency (see script comments)
- Direct coordinate computation using `element_ratios` for consistency

#### Lost/Gained Grid Plot

```bash
# Activate batch processing environment
source .venv311/bin/activate

# Generate 2x2 grid of Lost/Gained comparison plots
python scripts/plot_lost_gained_grid.py
```

Output: `data/comparisons_grid/lost_gained_grid.png`

Features:
- 4 pairwise comparisons: (1,2), (1,3), (1,4), (2,3)
- Unified axis limits for visual consistency
- Color schemes: Red-Green-Gray (default), Red-Blue, Orange-Purple
- Customizable marker sizes and transparency (see script comments)

**Note**: Both grid scripts include alternative styling options from previous versions as comments. Simply uncomment different sections to experiment with:
- Different color maps (`viridis`, `plasma`, `inferno`, `magma`, `cividis`)
- Point sizes (7, 10, 15, 20, 25, 35)
- Transparency levels (0.35, 0.5, 0.7, 0.85, 1.0)
- Edge styles (none, black with linewidth)

## Output Results

### Calculated Indices
- **Element Counts**: Number of C, H, N, O, P, S, Cl, F atoms
- **Element Ratios**: H/C, O/H, O/C ratios
- **DBE**: Double Bond Equivalence
- **rAI**: Modified Aromaticity Index
- **NOSC**: Nominal Oxidation State of Carbon
- **Mass Error**: Relative mass error (ppm)

### Visualization Charts
- Van Krevelen plots (DBE colored)
- Van Krevelen plots (density colored)
- Kendrick mass defect plots
- Atomic class distribution plots
- Compound class distribution plots
- Mass distribution histograms
- Mass spectra

## Dependency Conflict Resolution

**Problem**: pykrev 1.2.3 depends on the private function `_rot90_dispatcher` from `numpy.lib.function_base`, which was removed in NumPy 2.0+.

**Solutions**:
1. **Jupyter Environment**: Use runtime patches to inject missing functions before importing pykrev
2. **Batch Processing Environment**: Use the stable combination of Python 3.11 + NumPy 1.26.4

This dual-environment strategy ensures:
- Flexibility for development and interactive analysis (supports latest Python and NumPy)
- Stability for production batch processing (uses verified version combinations)

## Troubleshooting

### Virtual Environment Creation Failed
```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3.11-venv
```

### Excel Reading Error
Ensure `openpyxl` is installed and check that file format and column names are correct.

### pykrev Import Error
- Jupyter environment: Check if patch code is executed correctly
- Batch processing environment: Confirm using Python 3.11 and NumPy 1.26.4

## License

This project is for academic research use only.

## Acknowledgments

Provided materials serve only as reference examples; conclusions or analyses must be independently verified by the user.
