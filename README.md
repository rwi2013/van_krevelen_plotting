# FT-MS Data Processing Tools

This project provides Python tools for processing FT-MS (Fourier Transform Mass Spectrometry) data, including interactive Jupyter Notebooks and batch processing scripts.

Main parts of this code were informed by the following resources on preliminary FT-MS data processing:

- [FT-MS质谱数据的初步处理（文章）](https://mp.weixin.qq.com/s/UtmS3W1_LCmBhhLTyFFxPg)
- [FT-MS质谱数据的初步处理（视频）](https://www.bilibili.com/video/BV1SMomYwEs8)
Accessed on.

## Project Structure

```
├── scripts/
│   ├── FT-MS质谱数据的初步处理.ipynb  # Interactive analysis notebook
│   └── batch_export_ftms.py           # Batch processing script
├── data/                              # Data files
├── .venv/                             # Python 3.12 virtual environment
├── .venv311/                          # Python 3.11 virtual environment (for pykrev)
└── README.md
```

## Features

- **Molecular Formula Analysis**: Calculate element counts and ratios (H/C, O/H, O/C)
- **Chemical Indices**: Double Bond Equivalence (DBE), Aromaticity Index (rAI), Nominal Oxidation State of Carbon (NOSC)
- **Mass Calculations**: Nominal, average, monoisotopic, and theoretical masses
- **Mass Error**: Calculate relative mass error between measured and theoretical masses (ppm)
- **Kendrick Analysis**: Kendrick mass and mass defect calculations
- **Data Visualization**: Van Krevelen plots, mass spectra, atomic class distributions, etc.
- **Batch Processing**: Automated processing of multiple files with result export

## Data Format Requirements

Input files should be Excel format (.xlsx) containing the following required columns:
- `Formula`: Molecular formula (e.g., C6H12O6)
- `m/z`: Mass-to-charge ratio
- `Intensity`: Peak intensity

## Environment Setup

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
