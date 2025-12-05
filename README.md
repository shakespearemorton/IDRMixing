# Measuring (de)Mixing of Disordered Proteins through Domain Decomposition

[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://doi.org/PLACEHOLDER)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for reproducing the results in:

> **[Measuring (de)Mixing of Disordered Proteins through Domain Decomposition](https://doi.org/PLACEHOLDER)**  
> William Morton & Robert Vácha  
> *Journal Name*, 2025

## Overview

Biomolecular condensates organize cellular biochemistry through liquid-liquid phase separation (LLPS) driven by intrinsically disordered regions (IDRs). This work introduces a **domain decomposition method (DDM)** for quantifying phase concentrations and IDR mixing behavior in molecular dynamics simulations. We apply this methodology to 1,963 binary IDR mixtures to establish design rules for predicting whether IDRs will mix or segregate within condensates.

**Key contributions:**
- A voxel-based method for determining dense/dilute phase concentrations without fitting dividing surfaces
- A continuous **demixing index** (D) that characterizes spatial organization of multi-component condensates
- A logistic regression model for predicting demixing from sequence features
- Analysis of 496 unique IDR pairs across multiple concentration ratios

## Repository Structure

```
├── Analyze_Data/          # Core analysis scripts
├── Create_Subset/         # Dataset curation scripts
├── DATASETS/              # All input/output datasets
├── Figures/               # Figure generation scripts and data
├── MD/                    # Simulation input files
└── Run_MD/                # MD execution scripts
```

### `Analyze_Data/`
Scripts for analyzing simulation trajectories:
- **`ddm_homotypic.py`** — Domain decomposition analysis for single-component (homotypic) systems
- **`measure_demixing.py`** — Calculate the demixing index for binary mixtures

### `Create_Subset/`
Scripts for curating the IDR dataset:
- **`create.py`** — Select IDRs from the von Bülow dataset based on GIN clusters and ΔG values
- **`compositions.py`** — Generate concentration ratios for binary mixture simulations

### `DATASETS/`
CSV files containing sequences, simulation parameters, and results:

| File | Description |
|------|-------------|
| `gin_samples.csv` | 32 IDRs selected for this study with sequence features |
| `gin_prep.csv` | Simulation setup for all 1,963 binary mixtures |
| `demixing.csv` | Measured demixing indices and ΔG values |
| `df_training.csv` | Reference data from von Bülow et al. (216 IDRs) |
| `residues.csv` | Amino acid parameters (λ, charge, σ) for CALVADOS2 |
| `IDRome_DB_full.csv` | Full IDRome database for sequence lookups |
| `*_proteins.csv` | Protein metadata for experimental comparisons |
| `sabari_*.csv` | Data for comparison with Lyons et al. experiments |
| `exp_*.csv` / `known_*.csv` | Validation datasets |

### `Figures/`
Python scripts to reproduce all figures from the manuscript. Each script saves intermediate data to a corresponding `figure*_data/` directory for reproducibility.

| Script | Description |
|--------|-------------|
| `Figure1.py` | Domain decomposition method validation and mixing examples |
| `Figure2AC.py` | Composition-dependent demixing visualization |
| `Figure2GH.py` | Demixing index distributions |
| `Figure3.py` | Logistic regression model and condensate stability |
| `Figure4.py` | Experimental comparisons (FUS-A1, Lyons et al., Gilat et al.) |
| `Supplementary*.py` | Extended data figures |

The `figure3_data/` directory contains the trained demixing prediction model:
- `demixing_model_pipeline.joblib` — Sklearn pipeline (StandardScaler + LogisticRegression)
- `model_specification.json` — Feature definitions and raw coefficients
- `model_coefficients.csv` — Coefficient table

### `Run_MD/`
Scripts and templates for running CALVADOS2 direct coexistence simulations in OpenMM.

## Quick Start

### Predicting Demixing for New IDR Pairs

```python
from joblib import load
import pandas as pd

# Load the trained model
pipeline = load("Figures/figure3_data/demixing_model_pipeline.joblib")

# Calculate features for your IDR pair (see model_specification.json for definitions)
# Features: ncpr_mean, ncpr_abs_diff, faro_mean, faro_abs_diff, mean_lambda_mean, 
#           mean_lambda_abs_diff, N_mean, N_abs_diff, shd_mean, ...

features = ['ncpr_mean', 'ncpr_abs_diff', 'faro_abs_diff', 'N_mean', 
            'mean_lambda_mean', 'mean_lambda_abs_diff', 'shd_mean', 'N_abs_diff']

X = pd.DataFrame({
    'ncpr_mean': [(ncpr1 + ncpr2) / 2],
    'ncpr_abs_diff': [abs(ncpr1 - ncpr2)],
    # ... other features
})[features]

demixing_prob = pipeline.predict_proba(X)[:, 1]
```

### Analyzing Your Own Simulations

```python
from Analyze_Data.measure_demixing import calculate_demixing_index

# Load your trajectory with voxel-averaged concentrations
# Returns demixing index D ∈ [0, 1]
D = calculate_demixing_index(phi1_values, phi2_values, composition_ratio)
```

## Requirements

```
numpy>=1.26
pandas>=2.1
scipy>=1.13
scikit-learn>=1.5
matplotlib>=3.9
seaborn
MDAnalysis>=2.9
OpenMM>=8.2  # for running simulations
localcider   # for sequence feature calculation
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{morton2025mixing,
  title={Measuring (de)Mixing of Disordered Proteins through Domain Decomposition},
  author={Morton, William and V{\'a}cha, Robert},
  journal={},
  year={2025},
  doi={PLACEHOLDER}
}
```

## Related Resources

- [CALVADOS2 Force Field](https://github.com/KULL-Centre/CALVADOS)
- [von Bülow et al. Dataset](https://github.com/KULL-Centre/_2024_vonBuelow-et-al_LLPhD)
- [IDRome Database](https://github.com/KULL-Centre/_2023_Tesei_IDRome)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

William Morton — william.morton@ceitec.muni.cz  
CEITEC, Masaryk University, Brno, Czech Republic
